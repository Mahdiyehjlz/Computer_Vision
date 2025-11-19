import os
import cv2
import numpy as np
import signal
import sys

# Paths
DATA_PATH = '/Users/danish/PyTorch-YOLOv3/data/custom/images'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VALID_PATH = os.path.join(DATA_PATH, 'valid')
LABELS_FILE = os.path.join(DATA_PATH, 'bbox_labels.txt')
LOG_FILE = '/Users/danish/PyTorch-YOLOv3/output/log.txt'

# Handle KeyboardInterrupt
def signal_handler(sig, frame):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write("Process interrupted by user.\n")
    print("Process interrupted. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Read labels from text file
def load_labels(labels_file):
    labels_dict = {}
    with open(labels_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            img_name = parts[0]
            bbox = list(map(int, parts[1:5]))
            label = parts[5]
            if img_name not in labels_dict:
                labels_dict[img_name] = []
            labels_dict[img_name].append((bbox, label))
    return labels_dict

# Draw bounding boxes
def draw_bboxes(image, bboxes, labels):
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if label == 'dirt' else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_y = y1 - 10 if y1 >= 20 else y1 + 20
        cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

# Process images
def process_images(image_path, labels_dict):
    for img_name in os.listdir(image_path):
        img_path = os.path.join(image_path, img_name)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            key = os.path.splitext(img_name)[0]
            if key in labels_dict:
                bboxes_labels = labels_dict[key]
                bboxes = [item[0] for item in bboxes_labels]
                labels = [item[1] for item in bboxes_labels]
                image = draw_bboxes(image, bboxes, labels)
            output_path = os.path.join('/Users/danish/PyTorch-YOLOv3/output', img_name)
            os.makedirs('/Users/danish/PyTorch-YOLOv3/output', exist_ok=True)
            try:
                with open(LOG_FILE, 'a') as log_file:
                    log_file.write(f"Saving image to: {output_path}\n")
                    log_file.write(f"Image shape: {image.shape}\n")
                cv2.imwrite(output_path, image)
            except cv2.error as e:
                with open(LOG_FILE, 'a') as log_file:
                    log_file.write(f"Error saving {img_name}: {e}\n")


def main():
    labels_dict = load_labels(LABELS_FILE)
    with open(LOG_FILE, 'a') as log_file:
        log_file.write("Processing training images...\n")
    process_images(TRAIN_PATH, labels_dict)
    with open(LOG_FILE, 'a') as log_file:
        log_file.write("Processing validation images...\n")
    process_images(VALID_PATH, labels_dict)

if __name__ == "__main__":
    main()
