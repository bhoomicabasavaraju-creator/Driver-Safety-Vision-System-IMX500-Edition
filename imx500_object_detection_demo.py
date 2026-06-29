"""
Driver Safety Vision System — IMX500 Edition
============================================
Detects:
  1. Phone distraction      — IMX500 object detection
  2. Drowsiness (eyes)      — MediaPipe FaceMesh + EAR
  3. Sunglasses + no-move   — IMX500 sunglasses + MediaPipe Holistic nose tracking
                              → voice prompt → beep alert
"""

import argparse
import sys
import time
from functools import lru_cache

import cv2
import numpy as np
import mediapipe as mp
import subprocess
import os

from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection


# ──────────────────────────────────────────────
# AUDIO HELPERS
# ──────────────────────────────────────────────

def beep():
    """Non-blocking beep via paplay."""
    subprocess.Popen(
        ["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def speak(text: str):
    """Text-to-speech via espeak (blocking – runs fast)."""
    os.system(f'espeak -s 140 "{text}"')


# ──────────────────────────────────────────────
# CONSTANTS / THRESHOLDS
# ──────────────────────────────────────────────

EAR_THRESHOLD       = 0.20   # below → eyes considered closed
PHONE_TIME_LIMIT    = 2.0    # seconds phone near face → alert
EYE_TIME_LIMIT      = 2.0    # seconds eyes closed → drowsy alert
STILL_VOICE_DELAY   = 5.0    # seconds still with sunglasses → speak
POST_VOICE_DELAY    = 3.0    # seconds after voice, still still → beep
MOVEMENT_THRESHOLD  = 3      # pixel delta; below → "not moving"
PHONE_FACE_DIST     = 200    # max pixel dist for "phone near face"
PHONE_FACE_VDIFF    = 120    # max vertical diff for same

BEEP_COOLDOWN       = 2.0    # seconds between beeps while alert is ACTIVE

# FIX 2 — minimum box area (px²) to be considered the driver.
# Detections smaller than this are treated as background people.
# Tune this value for your camera distance / resolution.
MIN_DRIVER_BOX_AREA = 4000   # e.g. ~63×63 px minimum

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


# ──────────────────────────────────────────────
# GLOBALS SET DURING INIT
# ──────────────────────────────────────────────

imx500     = None
intrinsics = None
picam2     = None
args       = None

last_detections = []
last_results    = None


# ──────────────────────────────────────────────
# MEDIAPIPE SETUP
# ──────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_holistic = mp.solutions.holistic


# ──────────────────────────────────────────────
# EAR HELPERS
# ──────────────────────────────────────────────

def _dist(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def eye_aspect_ratio(eye_points) -> float:
    A = _dist(eye_points[1], eye_points[5])
    B = _dist(eye_points[2], eye_points[4])
    C = _dist(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0


# ──────────────────────────────────────────────
# BOX AREA HELPER
# ──────────────────────────────────────────────

def box_area(box) -> float:
    """Return pixel area of a (x1,y1,x2,y2) box."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


# ──────────────────────────────────────────────
# FIX 2 — LARGEST-BOX FILTER
# ──────────────────────────────────────────────

def largest_box(detections_of_class: list) -> tuple | None:
    """
    From a list of Detection objects of the same class, return the bounding
    box (x1,y1,x2,y2) of the one with the LARGEST area — that is the person
    closest to the camera (the driver).  Returns None if the list is empty or
    if the biggest box is smaller than MIN_DRIVER_BOX_AREA (background person).
    """
    if not detections_of_class:
        return None

    best = max(
        detections_of_class,
        key=lambda d: box_area(_det_to_xyxy(d)),
    )
    b = _det_to_xyxy(best)
    if box_area(b) < MIN_DRIVER_BOX_AREA:
        return None          # too small → background, ignore
    return b


def _det_to_xyxy(det) -> tuple:
    x, y, bw, bh = det.box
    return (x, y, x + bw, y + bh)


# ──────────────────────────────────────────────
# PHONE ↔ FACE PROXIMITY
# ──────────────────────────────────────────────

def is_phone_near_face(face_box, phone_box) -> bool:
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = phone_box
    fc = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
    pc = ((px1 + px2) / 2, (py1 + py2) / 2)
    dist  = np.linalg.norm(np.array(fc) - np.array(pc))
    vdiff = abs(fc[1] - pc[1])
    return dist < PHONE_FACE_DIST and vdiff < PHONE_FACE_VDIFF


# ──────────────────────────────────────────────
# IMX500 DETECTION CLASS
# ──────────────────────────────────────────────

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf     = conf
        self.box      = imx500.convert_inference_coords(coords, metadata, picam2)


# ──────────────────────────────────────────────
# PARSE IMX500 DETECTIONS
# ──────────────────────────────────────────────

def parse_detections(metadata: dict):
    global last_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections

    input_w, input_h = imx500.get_input_size()
    threshold       = args.threshold
    iou             = args.iou
    max_detections  = args.max_detections

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=threshold,
            iou_thres=iou,
            max_out_dets=max_detections,
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if intrinsics.bbox_normalization:
            boxes = boxes / input_h
        if intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_detections


# ──────────────────────────────────────────────
# LABELS + CLASS INDEX LOOKUP
# ──────────────────────────────────────────────

@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [l for l in labels if l and l != "-"]
    return labels


def get_class_indices():
    """Return (phone_idx, face_idx, sunglasses_idx) from label list."""
    labels = get_labels()
    phone_idx = face_idx = sunglasses_idx = None
    for i, lab in enumerate(labels):
        l = lab.lower()
        if "cell phone" in l or ("phone" in l and phone_idx is None):
            phone_idx = i
        if "face" in l and face_idx is None:
            face_idx = i
        if "sunglass" in l and sunglasses_idx is None:
            sunglasses_idx = i
    return phone_idx, face_idx, sunglasses_idx


# ──────────────────────────────────────────────
# OVERLAY HELPERS
# ──────────────────────────────────────────────

def put_alert(img, text: str, y: int, color=(0, 0, 255), scale=1.0, thick=3):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def put_info(img, text: str, y: int, color=(200, 200, 255), scale=0.75, thick=2):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_box(img, x1, y1, x2, y2, color=(0, 255, 0), label=""):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1 + 4, y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ──────────────────────────────────────────────
# ARGUMENT PARSING
# ──────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Driver Safety Vision System")
    parser.add_argument(
        "--model", type=str,
        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
    )
    parser.add_argument("--fps",              type=int)
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--bbox-order",       choices=["yx", "xy"], default="yx")
    parser.add_argument("--threshold",        type=float, default=0.20)
    parser.add_argument("--iou",              type=float, default=0.65)
    parser.add_argument("--max-detections",   type=int,   default=10)
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction)
    parser.add_argument("--postprocess",      choices=["", "nanodet"], default=None)
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction)
    parser.add_argument("--labels",           type=str)
    parser.add_argument("--print-intrinsics", action="store_true")
    return parser.parse_args()


# ──────────────────────────────────────────────
# STATE CLASS
# ──────────────────────────────────────────────

class SafetyState:
    def __init__(self):
        # Phone
        self.phone_start: float | None = None

        # Drowsiness (EAR)
        self.eye_start: float | None = None

        # Sunglasses + no-movement
        self.prev_nose: tuple | None     = None
        self.no_move_start: float | None = None
        self.voice_played: bool          = False
        self.voice_play_time: float | None = None

        # FIX 1 — beep cooldown per alert channel.
        # Each channel has its own last_beep timestamp.
        # When the alert condition clears we reset that channel's
        # last_beep to 0.0, so the NEXT genuine alert fires instantly
        # and there is no carry-over beep after the condition ends.
        self.last_beep_phone:   float = 0.0
        self.last_beep_eye:     float = 0.0
        self.last_beep_sunglass: float = 0.0

    # ── per-channel beep helpers ──────────────

    def _try_beep(self, channel: str) -> None:
        attr = f"last_beep_{channel}"
        now  = time.time()
        if now - getattr(self, attr) >= BEEP_COOLDOWN:
            beep()
            setattr(self, attr, now)

    def _reset_beep(self, channel: str) -> None:
        """
        FIX 1 — Call this the moment an alert condition clears.
        Resetting to 0.0 means the cooldown timer is instantly expired,
        so the next alert fires its beep immediately with no stale delay.
        """
        setattr(self, f"last_beep_{channel}", 0.0)

    def phone_beep(self):      self._try_beep("phone")
    def eye_beep(self):        self._try_beep("eye")
    def sunglass_beep(self):   self._try_beep("sunglass")

    def phone_clear(self):     self._reset_beep("phone")
    def eye_clear(self):       self._reset_beep("eye")
    def sunglass_clear(self):  self._reset_beep("sunglass")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    global imx500, intrinsics, picam2, args, last_results

    args = get_args()

    # ── IMX500 init ──────────────────────────
    imx500     = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    if not imx500.network_intrinsics:
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Model is not an object-detection network.", file=sys.stderr)
        sys.exit(1)

    for key, value in vars(args).items():
        if key == "labels" and value is not None:
            with open(value) as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    if intrinsics.labels is None:
        with open("assets/coco_labels.txt") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        sys.exit(0)

    # ── Camera init ───────────────────────────
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12,
    )
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    # ── Class index lookup ────────────────────
    phone_idx, face_idx, sunglasses_idx = get_class_indices()
    print(f"Labels : {get_labels()}")
    print(f"phone={phone_idx}  face={face_idx}  sunglasses={sunglasses_idx}")

    # ── MediaPipe Holistic ────────────────────
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
    )

    state = SafetyState()

    # ── Main loop ─────────────────────────────
    while True:
        # --- capture frame + detections ---
        metadata     = picam2.capture_metadata()
        last_results = parse_detections(metadata)

        frame = picam2.capture_array("main")          # RGB
        bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w  = bgr.shape[:2]
        annotated = bgr.copy()

        # ══════════════════════════════════════
        # FIX 2 — DRIVER SELECTION
        # Bucket detections by class, then keep only the
        # LARGEST bounding box per class (= closest to camera).
        # Small background detections are discarded automatically.
        # ══════════════════════════════════════
        phone_dets      = []
        face_dets       = []
        sunglass_dets   = []

        if last_results:
            for det in last_results:
                cls = int(det.category)
                if phone_idx      is not None and cls == phone_idx:
                    phone_dets.append(det)
                elif face_idx     is not None and cls == face_idx:
                    face_dets.append(det)
                elif sunglasses_idx is not None and cls == sunglasses_idx:
                    sunglass_dets.append(det)

                # Draw ALL raw detections in grey so you can still see them
                x, y, bw, bh = det.box
                label_name = get_labels()[cls] if cls < len(get_labels()) else str(cls)
                draw_box(annotated, x, y, x + bw, y + bh,
                         (100, 100, 100), f"{label_name} {det.conf:.2f}")

        # Pick only the largest (= nearest) box per class
        phone_box           = largest_box(phone_dets)
        face_box            = largest_box(face_dets)
        sunglasses_detected = largest_box(sunglass_dets) is not None

        # Re-draw the driver-selected boxes in bright colours
        if phone_box:
            draw_box(annotated, *phone_box[:2], *phone_box[2:], (0, 128, 255), "PHONE(driver)")
        if face_box:
            draw_box(annotated, *face_box[:2], *face_box[2:], (0, 255, 0), "FACE(driver)")

        # ══════════════════════════════════════
        # 1. PHONE DISTRACTION
        # ══════════════════════════════════════
        phone_alert = False
        if phone_box is not None:
            if face_box is None or is_phone_near_face(face_box, phone_box):
                phone_alert = True

        if phone_alert:
            if state.phone_start is None:
                state.phone_start = time.time()
            elapsed = time.time() - state.phone_start
            put_info(annotated, f"Phone detected: {elapsed:.1f}s", 40, (0, 180, 255))
            if elapsed >= PHONE_TIME_LIMIT:
                put_alert(annotated, "!! PHONE DISTRACTION!", 85)
                state.phone_beep()
        else:
            # FIX 1 — condition cleared: reset timer AND beep cooldown
            if state.phone_start is not None:
                state.phone_clear()       # ← beep stops immediately
            state.phone_start = None

        # ══════════════════════════════════════
        # 2. DROWSINESS — EAR (skip if sunglasses on)
        # ══════════════════════════════════════
        if face_box is not None and not sunglasses_detected:
            rgb_frame    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mesh_results = face_mesh.process(rgb_frame)

            if mesh_results.multi_face_landmarks:
                lm = mesh_results.multi_face_landmarks[0].landmark

                left_eye  = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE_IDX]
                right_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE_IDX]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                put_info(annotated, f"EAR: {ear:.3f}", 130, (180, 180, 255))

                if ear < EAR_THRESHOLD:
                    if state.eye_start is None:
                        state.eye_start = time.time()
                    elapsed = time.time() - state.eye_start
                    put_info(annotated, f"Eyes closed: {elapsed:.1f}s", 165, (0, 140, 255))
                    if elapsed >= EYE_TIME_LIMIT:
                        put_alert(annotated, "!! DROWSINESS DETECTED!", 210)
                        state.eye_beep()
                else:
                    # FIX 1 — eyes opened: stop beeping immediately
                    if state.eye_start is not None:
                        state.eye_clear()     # ← beep stops immediately
                    state.eye_start = None
            else:
                # Face mesh lost — reset
                if state.eye_start is not None:
                    state.eye_clear()
                state.eye_start = None

        elif sunglasses_detected:
            if state.eye_start is not None:
                state.eye_clear()
            state.eye_start = None

        # ══════════════════════════════════════
        # 3. SUNGLASSES + HEAD STILLNESS
        # ══════════════════════════════════════
        if sunglasses_detected:
            put_info(annotated, "Sunglasses detected", 255, (0, 255, 255))

        rgb_frame        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        holistic_results = holistic.process(rgb_frame)

        if sunglasses_detected and holistic_results.pose_landmarks:
            nose_lm = holistic_results.pose_landmarks.landmark[
                mp_holistic.PoseLandmark.NOSE
            ]
            nose_pt = (int(nose_lm.x * w), int(nose_lm.y * h))

            if state.prev_nose is not None:
                movement = np.linalg.norm(np.array(nose_pt) - np.array(state.prev_nose))

                if movement < MOVEMENT_THRESHOLD:
                    if state.no_move_start is None:
                        state.no_move_start = time.time()
                    still_time = time.time() - state.no_move_start

                    put_info(annotated, f"No movement: {still_time:.1f}s", 295, (0, 220, 220))

                    # Stage 1 → voice prompt
                    if still_time >= STILL_VOICE_DELAY and not state.voice_played:
                        speak("Hello, how are you doing?")
                        state.voice_played    = True
                        state.voice_play_time = time.time()

                    # Stage 2 → beep alert
                    if state.voice_played and state.voice_play_time is not None:
                        post_voice = time.time() - state.voice_play_time
                        put_info(annotated,
                                 f"Post-voice: {post_voice:.1f}s", 330, (0, 180, 180))
                        if post_voice >= POST_VOICE_DELAY:
                            put_alert(annotated, "!! SUNGLASSES DROWSINESS!", 375)
                            state.sunglass_beep()

                else:
                    # FIX 1 — head moved: stop sunglass beep immediately
                    if state.no_move_start is not None:
                        state.sunglass_clear()    # ← beep stops immediately
                    state.no_move_start   = None
                    state.voice_played    = False
                    state.voice_play_time = None

            state.prev_nose = nose_pt

        else:
            # Sunglasses gone or pose lost — full reset
            if state.no_move_start is not None:
                state.sunglass_clear()
            state.no_move_start   = None
            state.voice_played    = False
            state.voice_play_time = None
            state.prev_nose       = None

        # ══════════════════════════════════════
        # STATUS BAR
        # ══════════════════════════════════════
        status_items = []
        if phone_alert:
            status_items.append("PHONE")
        if sunglasses_detected:
            status_items.append("SUNGLASS")
        if face_box is not None:
            status_items.append("FACE")

        status_text  = " | ".join(status_items) if status_items else "CLEAR"
        status_color = (0, 255, 0) if status_text == "CLEAR" else (0, 165, 255)
        text_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(annotated, status_text,
                    (w - text_size[0] - 15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # ══════════════════════════════════════
        # DISPLAY
        # ══════════════════════════════════════
        cv2.imshow("Driver Safety Vision System", annotated)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC → quit
            break

    # ── Cleanup ───────────────────────────────
    picam2.stop()
    holistic.close()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()