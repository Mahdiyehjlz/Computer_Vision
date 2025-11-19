#!/usr/bin/env python3
import os, argparse
from collections import defaultdict
from PIL import Image

# ←––– Update this list to match the exact order in your data.yaml
CLASS_NAMES = [
    "business_cards",
    "dirt",
    "keys",
    "other",
    "paper_and_notebooks",
    "pens",
    "rulers",
    "scissors",
    "tapes",
    "usb_sticks"
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--labels',  required=True, help='path to bbox_labels.txt')
    p.add_argument('--imgroot', required=True, help='images root, with train/ and valid/')
    p.add_argument('--out',     required=True, help='output labels root (will have train/ & valid/)')
    args = p.parse_args()

    # 1) Read the single big file into memory
    annots = defaultdict(list)
    with open(args.labels) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            img_id, x1, y1, x2, y2, cls_name = parts
            try:
                cid = CLASS_NAMES.index(cls_name)
            except ValueError:
                # unknown class—skip or log
                print(f"⚠️  Warning: '{cls_name}' not in CLASS_NAMES, skipping")
                continue
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            annots[img_id].append((cid, x1, y1, x2, y2))

    # 2) Process both splits
    for split in ('train', 'valid'):
        img_dir = os.path.join(args.imgroot, split)
        out_dir = os.path.join(args.out,    split)
        os.makedirs(out_dir, exist_ok=True)

        for img_name in os.listdir(img_dir):
            base = os.path.splitext(img_name)[0]
            if base not in annots:
                continue

            w, h = Image.open(os.path.join(img_dir, img_name)).size

            with open(os.path.join(out_dir, base + '.txt'), 'w') as out:
                for cid, x1, y1, x2, y2 in annots[base]:
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    out.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print("✅ Done splitting labels into 10 classes.")

if __name__ == '__main__':
    main()
