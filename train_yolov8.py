#!/usr/bin/env python3
from ultralytics import YOLO

def train_dirtnet_largest():
    model = YOLO("yolov8x.pt")
    model.train(
        data="data/dirt.yaml",
        device="0,1",
        workers=8,
        cache="disk",
        amp=True,
        imgsz=1024,
        batch=16,
        epochs=50,
        patience=10,
        lr0=0.001,
        lrf=0.1,
        warmup_epochs=5,
        mosaic=1.0,
        copy_paste=0.5,
        mixup=0.2,
        auto_augment="randaugment",
        project="runs/train",
        name="dirtnet_largest"
    )

if __name__ == "__main__":
    train_dirtnet_largest()
