#!/usr/bin/env python3
import sys
import os
from ultralytics import YOLO

def train_dirtnet_last():
    model = YOLO("yolo11x.pt")
    model.train(
        data="data/dirt.yaml",
        device="0,1",
        workers=8,
        cache="disk",
        amp=True,
        imgsz=1024,
        batch=16,
        epochs=300,
        patience=10,
        lr0=0.001,
        lrf=0.1,
        warmup_epochs=5,
        mosaic=1.0,
        copy_paste=0.5,
        mixup=0.2,
        auto_augment="randaugment",
        project="runs/train",
        name="dirtnet_last"
    )

if __name__ == "__main__":
    # Ensure log directory exists
    log_dir = Path("runs/train/dirtnet_last")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    # Redirect stdout and stderr to log file
    with open(log_file, "w") as f:
        sys.stdout = f
        sys.stderr = f
        train_dirtnet_last()

# to track use:
#>> tail -v -f runs/train/dirtnet_last/train.log
# Now let's say you want to lose connection to remote computer, go to work and
# let it train in background
# >> jobs -l
#(you need to remember the job number to later use it in ps - p <JobNumber>..) 
#say it shows [1]+ 465201 Running …  (& pid 465201)
# >> disown %1
#ps -p 465201 -o pid,ppid,cmd
#If after the last command you dont see anything, you have succesfully disowned the
# the process and now you can turn off your laptop i.e lose connection to remote computer
# and it will run in background
        
