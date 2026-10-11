from argparse import ArgumentParser
from ultralytics import YOLO


def main():
    parser = ArgumentParser(description="YOLO Model Exporter")

    parser.add_argument(
        "--model",
        required=True,
        help="Path to .pt model"
    )

    parser.add_argument(
        "--format",
        required=True,
        choices=["onnx", "openvino", "engine"],
        help="Export format"
    )

    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 when supported"
    )

    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input size"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size"
    )

    args = parser.parse_args()

    model = YOLO(args.model)

    model.export(
        format=args.format,
        half=args.half,
        dynamic=args.dynamic,
        imgsz=args.imgsz
    )


if __name__ == "__main__":
    main()