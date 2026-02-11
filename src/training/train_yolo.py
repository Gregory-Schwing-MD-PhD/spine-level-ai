#!/usr/bin/env python3
"""
Train YOLOv11 on LSTV detection with WandB logging
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import wandb
import json
from datetime import datetime

def train_yolo(data_yaml, model_size='n', epochs=100, batch_size=16, 
               img_size=640, project_dir='runs/lstv', name='trial',
               resume=False, wandb_logging=True):
    """Train YOLOv11 model"""
    
    config = {
        'yolo_version': '11',
        'model_size': model_size,
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': img_size,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }
    
    if wandb_logging:
        wandb.init(
            project='lstv-detection',
            name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
            save_code=True,
        )
    
    model_name = f'yolo11{model_size}.pt'
    print(f"\nLoading {model_name}...")
    print(f"Using YOLOv11:")
    print(f"  - 22% fewer parameters than YOLOv8")
    print(f"  - Better small object detection")
    print(f"  - Faster training\n")
    
    model = YOLO(model_name)
    
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'project': project_dir,
        'name': name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': resume,
        'amp': True,
        'lr0': config['lr0'],
        'lrf': config['lrf'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'warmup_epochs': config['warmup_epochs'],
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': config['box'],
        'cls': config['cls'],
        'dfl': config['dfl'],
        'val': True,
        'plots': True,
        'save': True,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'patience': 50,
    }
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("="*60)
    
    print("\nStarting training...")
    results = model.train(**train_args)
    
    print("\nValidating...")
    metrics = model.val()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best mAP50: {metrics.box.map50:.4f}")
    print(f"Best mAP50-95: {metrics.box.map:.4f}")
    print(f"Model saved to: {project_dir}/{name}/weights/best.pt")
    print("="*60)
    
    metrics_dict = {
        'yolo_version': 11,
        'map50': float(metrics.box.map50),
        'map50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'config': config,
    }
    
    metrics_path = Path(project_dir) / name / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    if wandb_logging:
        wandb.log(metrics_dict)
        wandb.finish()
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 on LSTV detection')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--project', type=str, default='runs/lstv')
    parser.add_argument('--name', type=str, default='train')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: No GPU detected, training will be slow!")
    
    model, metrics = train_yolo(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        project_dir=args.project,
        name=args.name,
        resume=args.resume,
        wandb_logging=not args.no_wandb,
    )
    
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()
