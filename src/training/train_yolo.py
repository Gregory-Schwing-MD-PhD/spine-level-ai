"""
YOLOv8 Training Script for Spine Level Identification
"""

import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="models")
    parser.add_argument("--name", type=str, default="spine_level_v1")
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    model = YOLO('yolov8m.pt')
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img,
        device=args.device,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        patience=args.patience,
        workers=args.workers
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
