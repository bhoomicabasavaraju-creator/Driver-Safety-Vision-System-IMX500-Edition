# Driver Safety Vision System — IMX500 Edition

A real-time driver monitoring system running on a Raspberry Pi with the Sony IMX500 AI camera. It detects three safety-critical conditions and alerts the driver via audio beeps and on-screen overlays.

---

## Features

| Detection | Method | Alert |
|---|---|---|
| Phone distraction | IMX500 object detection (COCO cell phone class) | Beep after 2s |
| Drowsiness (eyes closed) | MediaPipe FaceMesh + Eye Aspect Ratio (EAR) | Beep after 2s |
| Sunglasses + head stillness | IMX500 sunglasses class + MediaPipe Holistic nose tracking | Voice prompt → beep |

---

## Hardware Requirements

- Raspberry Pi (tested with Pi 5)
- [Raspberry Pi AI Camera (IMX500)](https://www.raspberrypi.com/products/ai-camera/)
- Speaker or audio output (for beep/TTS alerts)

---

## Software Requirements

- Python 3.10 (required for IMX500 toolchain)
- Picamera2
- MediaPipe
- OpenCV
- `espeak` (text-to-speech)
- `paplay` / PulseAudio (beep alerts)

---

## Installation

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install imx500-all espeak pulseaudio
```

### 2. Set up the Python environment

```bash
# Install uv package manager
wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create project
mkdir driver-safety && cd driver-safety
uv init .

# Create venv with access to system site-packages (required for Picamera2)
uv venv --system-site-packages

# Install dependencies
uv pip install "numpy<2" opencv-python-headless mediapipe
```

### 3. Clone / copy the script

Place `imx500_object_detection_demo.py` in your project directory.

---


## How It Works

### 1. Phone Distraction
The IMX500 detects `cell phone` objects. If a phone is detected near the driver's face (within `PHONE_FACE_DIST=200px` horizontally and `PHONE_FACE_VDIFF=120px` vertically) for more than **2 seconds**, a beep alert fires.

### 2. Drowsiness — Eye Aspect Ratio (EAR)
MediaPipe FaceMesh extracts 6 landmarks per eye. The Eye Aspect Ratio is computed as:

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

If EAR drops below `0.20` for more than **2 seconds**, a drowsiness alert fires. This check is skipped when sunglasses are detected.

### 3. Sunglasses + Head Stillness
When the IMX500 detects sunglasses, EAR-based eye tracking is unreliable. Instead, the system tracks the nose landmark from MediaPipe Holistic:
- If the nose moves less than **3 pixels** between frames for **5 seconds** → a voice prompt plays: *"Hello, how are you doing?"*
- If no response (still no movement) for **3 more seconds** → beep alert fires

### Driver Selection (Multi-Person Robustness)
When multiple people are in frame, the system uses the **largest bounding box per class** as the driver (closest to camera). Detections smaller than `MIN_DRIVER_BOX_AREA = 4000 px²` are treated as background.

### Beep Cooldown
Each alert channel (phone, eye, sunglasses) has an independent **2-second beep cooldown**. When an alert condition clears, the cooldown resets immediately so the next genuine alert fires without delay.

---

## Key Thresholds (tunable in code)

```python
EAR_THRESHOLD       = 0.20   # EAR below this → eyes closed
PHONE_TIME_LIMIT    = 2.0    # seconds with phone near face → alert
EYE_TIME_LIMIT      = 2.0    # seconds eyes closed → drowsy alert
STILL_VOICE_DELAY   = 5.0    # seconds still with sunglasses → voice prompt
POST_VOICE_DELAY    = 3.0    # seconds after voice, still still → beep
MOVEMENT_THRESHOLD  = 3      # pixel delta below this → "not moving"
PHONE_FACE_DIST     = 200    # max pixel distance for "phone near face"
MIN_DRIVER_BOX_AREA = 4000   # min box area (px²) to count as driver
BEEP_COOLDOWN       = 2.0    # seconds between repeated beeps
```

---

## Project Structure

```
driver-safety/
├── imx500_object_detection_demo.py       # Main inference script (Raspberry Pi)
├── aisd_model_training.ipynb         # Custom model training script (Google Colab)
├── data sets/
│   └── all_images.zip
│   └── all_images.zip
└── README.md
```

---

## Training a Custom Model

The custom model was trained on Google Colab using `aisd_model_training.ipynb`. It fine-tunes **YOLOv8n** on a dataset of 3 domain-specific classes: `phone`, `face`, and `sunglasses`.

### Classes

| ID | Name | Notes |
|---|---|---|
| 0 | phone | Cell phone held near face |
| 1 | face | Driver's face |
| 2 | sunglasses | Sunglasses worn by driver |

### Step-by-step

**1. Prepare your data**

Annotate images using Label Studio and export in **YOLO with Images** format. Then zip images and labels separately:

```
all_images.zip   ← all annotated images
all_labels.zip   ← matching YOLO .txt label files
```

**2. Open `aisd_model_training.ipynb` in Google Colab**

When prompted, upload `all_images.zip` and `all_labels.zip`. The script will:

- Extract both zips into `/content/images_raw` and `/content/labels_raw`
- **Oversample sunglasses images ×8** to compensate for class imbalance (sunglasses are rare in typical datasets)
- Shuffle and split the final pool into **80% train / 10% val / 10% test**
- Print a class distribution report so you can verify balance before training

**3. Training configuration**

The model trains for up to **200 epochs** with early stopping (`patience=50`):

```python
model = YOLO("yolov8n.pt")

model.train(
    data    = "data.yaml",
    epochs  = 200,
    imgsz   = 640,
    batch   = 16,
    patience= 50,
    workers = 2,

    # Augmentation
    degrees    = 10,    # rotation
    translate  = 0.05,  # translation
    scale      = 0.5,   # scale jitter
    shear      = 2,     # shear
    fliplr     = 0.5,   # horizontal flip
    mosaic     = 0.7,   # mosaic augmentation
    mixup      = 0.15,  # mixup
    copy_paste = 0.4,   # copy-paste
)
```

**4. Output**

Training results are saved to `/content/DSVS_training/driver_monitoring/`. The best weights are automatically downloaded at the end:

```
best.pt   ← use this for IMX500 export
```

**5. Export to IMX500 (Linux only, Python 3.10)**

On the Linux workstation, convert `best.pt` to an IMX500-compatible model:
```bash
uv run python yolo_export.py \
  --init_model best.pt \
  --export_format imx \
  --export_only \
  --int8_weights
```

This produces `best_imx_model/packerOut.zip` and `best_imx_model/labels.txt`.

**6. Package for Raspberry Pi**

Copy `packerOut.zip` and `labels.txt` to the Pi, then run:

```bash
sudo apt install imx500-all
imx500-package -i packerOut.zip -o ./model_output
# Output: model_output/network.rpk
```

**7. Run on Raspberry Pi**

```bash
uv run python imx500_object_detection_demo.py \
  --model ./model_output/network.rpk \
  --labels ./best_imx_model/labels.txt \
  --bbox-normalization \
  --bbox-order xy
```
### All arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | SSD MobileNetV2 FPN (bundled) | Path to `.rpk` model file |
| `--labels` | `assets/coco_labels.txt` | Path to labels text file |
| `--threshold` | `0.20` | Detection confidence threshold |
| `--iou` | `0.65` | NMS IoU threshold |
| `--max-detections` | `10` | Max detections per frame |
| `--fps` | (from model intrinsics) | Camera frame rate |
| `--bbox-normalization` | — | Enable bounding box normalization |
| `--bbox-order` | `yx` | Bounding box coordinate order (`yx` or `xy`) |
| `--postprocess` | — | Postprocess mode (`nanodet` or empty) |
| `--preserve-aspect-ratio` | — | Preserve aspect ratio on inference input |
| `--print-intrinsics` | — | Print model intrinsics and exit |

---

> The IMX500 AI camera supports **YOLOv8n** and **YOLO11n** only.

---

## Acknowledgements

- [Picamera2](https://github.com/raspberrypi/picamera2) — camera interface and IMX500 integration
- [MediaPipe](https://google.github.io/mediapipe/) — FaceMesh and Holistic landmark detection
- [Ultralytics YOLO](https://docs.ultralytics.com/) — model training and export
- [Sony IMX500](https://developer.sony.com/imx500) — on-sensor AI inference
- Project guidance by Prof. Dr. Thomas Ewender, Deggendorf Institute of Technology
