from ultralytics import YOLO
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO trainer")
    parser.add_argument("--config", type=str, default="config.yaml", help="The dataset config file")
    parser.add_argument("--init_model", type=str, default="yolo11n.pt", help="The pre-trained model weights")

    parser.add_argument("--name", type=str, default="yolo", help="Save dir")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--device", type=int, default=0, help="Device training index")
    parser.add_argument("--gpu_percent", type=float, default=0.9, help="How much of the GPU RAM to use")

    parser.add_argument("--export_format", type=str, default="onnx", help="format to export the model as, eg onnx, imx")
    parser.add_argument("--export_config", type=str, default=None, help="The dataset config file for export")
    parser.add_argument("--resume_training",  action='store_true', help="Resume training of a model")
    parser.add_argument("--export_only",  action='store_true', help="Just export the weights as onnx")
    parser.add_argument("--int8_weights",  action='store_true', help="Export the weights as int8")
    parser.add_argument("--image_size", type=str, default="640x640",
                        help="Image size as width height (default: 640x640)")
    parser.add_argument("--val_model",  action='store_true', help="Validate the model only")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load a YOLOv8n PyTorch model
    model = YOLO(args.init_model)

    image_h, image_w = map(int, args.image_size.split('x'))
    image_size = [image_h, image_w]
    print(image_size)

    if not args.export_only and not args.val_model:
        if args.resume_training:
            project = args.name
        else:
            project = None

        _ = model.train(data=args.config, epochs=args.epochs, imgsz=image_size, save=True,
                              device=args.device, name=args.name, batch=args.gpu_percent, resume=args.resume_training,
                              cache=False, project=project)
    elif  args.val_model:
        _ = model.val(name=args.name, project=args.name)  # no arguments needed, dataset and settings remembered

    # Export the model
    if args.export_format:
        export_args = {
            "format": args.export_format,
            "int8": args.int8_weights,
            "imgsz": image_size,
            "device": "cpu",
            "nms": True,
            "data": args.export_config,
            "opset": 11,
            "name": args.name,
            "project": args.name
        }

    # Remove unsupported arguments for 'imx' format
    if args.export_format != "imx":
        export_args["batch"] = 16

    model.export(**export_args)
if __name__ == "__main__":
    main()
