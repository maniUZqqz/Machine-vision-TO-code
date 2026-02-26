#!/usr/bin/env python
"""
Auto-labeler: Use the existing CV pipeline to generate YOLO-format labels
for UI element detection training.

Generates:
  - Copies images to dataset/images/train/
  - Creates .txt label files in dataset/labels/train/
  - Each line: class_id cx cy w h (normalized 0-1)

Classes:
  0: button
  1: input
  2: select
  3: card          (container with distinct bg)
  4: image         (photo / embedded image)
  5: icon
  6: text_block    (heading or paragraph)
  7: header_bar    (full-width top bar)
  8: sidebar       (tall narrow side panel)
  9: navbar
  10: separator

Usage:
    python auto_labeler.py [image_dir] [--output dataset/]
"""
import os
import sys
import shutil
import argparse

# Setup Django
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

import numpy as np
from PIL import Image
from vision_engine.pipeline import create_default_pipeline
from vision_engine.models.elements import ElementType


# Map ElementType + semantic_tag → YOLO class
CLASS_MAP = {
    'button': 0,
    'input': 1,
    'select': 2,
    'card': 3,
    'image': 4,
    'icon': 5,
    'text_block': 6,
    'header_bar': 7,
    'sidebar': 8,
    'navbar': 9,
    'separator': 10,
}

CLASS_NAMES = list(CLASS_MAP.keys())


def element_to_class(elem):
    """Map a DetectedElement to a YOLO class ID."""
    etype = elem.element_type

    if etype == ElementType.BUTTON:
        return CLASS_MAP['button']
    if etype == ElementType.INPUT:
        return CLASS_MAP['input']
    if etype == ElementType.SELECT:
        return CLASS_MAP['select']
    if etype == ElementType.CHECKBOX or etype == ElementType.RADIO:
        return CLASS_MAP['input']
    if etype == ElementType.TEXTAREA:
        return CLASS_MAP['input']
    if etype == ElementType.SEPARATOR:
        return CLASS_MAP['separator']

    if etype == ElementType.IMAGE:
        if getattr(elem, 'is_icon', False):
            return CLASS_MAP['icon']
        return CLASS_MAP['image']

    if etype == ElementType.TEXT:
        return CLASS_MAP['text_block']

    # Container — classify by semantic tag
    tag = getattr(elem, 'semantic_tag', 'div')
    if tag == 'header':
        return CLASS_MAP['header_bar']
    if tag == 'aside':
        return CLASS_MAP['sidebar']
    if tag == 'nav':
        return CLASS_MAP['navbar']
    if tag == 'footer':
        return CLASS_MAP['header_bar']  # footer ~ header_bar visually

    # Generic container with visible bg → card
    if elem.color and elem.color.background:
        return CLASS_MAP['card']

    return CLASS_MAP['card']


def generate_labels(image_path, pipeline):
    """Run pipeline and generate YOLO label lines."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    ctx = pipeline.run(img_array)

    labels = []

    def walk(elements):
        for elem in elements:
            cls_id = element_to_class(elem)
            b = elem.bbox

            # Normalize to 0-1
            cx = (b.x + b.width / 2) / w
            cy = (b.y + b.height / 2) / h
            nw = b.width / w
            nh = b.height / h

            # Clamp to valid range
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0.001, min(1, nw))
            nh = max(0.001, min(1, nh))

            # Skip tiny annotations
            if b.area < 200:
                walk(elem.children)
                continue

            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            walk(elem.children)

    walk(ctx.regions)

    # Also add text elements
    for t in ctx.text_elements:
        cls_id = CLASS_MAP['text_block']
        b = t.bbox
        cx = (b.x + b.width / 2) / w
        cy = (b.y + b.height / 2) / h
        nw = b.width / w
        nh = b.height / h

        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        nw = max(0.001, min(1, nw))
        nh = max(0.001, min(1, nh))

        if b.area >= 100:
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return labels


def main():
    parser = argparse.ArgumentParser(description='Auto-label images for YOLO training')
    parser.add_argument('image_dir', nargs='?', default='../../test_samples/input',
                        help='Directory with input images')
    parser.add_argument('--output', default='dataset',
                        help='Output dataset directory')
    parser.add_argument('--split', type=float, default=0.8,
                        help='Train/val split ratio')
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, args.image_dir) if not os.path.isabs(args.image_dir) else args.image_dir
    output_dir = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output

    # Find images
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    images = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ]

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images in {image_dir}")
    print("Initializing pipeline...")
    pipeline = create_default_pipeline()
    print("Pipeline ready.\n")

    # Create output dirs
    for split in ('train', 'val'):
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Process each image
    import random
    random.shuffle(images)
    split_idx = int(len(images) * args.split)

    for i, img_path in enumerate(images):
        split = 'train' if i < split_idx else 'val'
        basename = os.path.splitext(os.path.basename(img_path))[0]

        print(f"  [{i+1}/{len(images)}] {os.path.basename(img_path)} -> {split} ... ", end='', flush=True)

        try:
            labels = generate_labels(img_path, pipeline)

            # Copy image
            dst_img = os.path.join(output_dir, 'images', split, os.path.basename(img_path))
            shutil.copy2(img_path, dst_img)

            # Write labels
            dst_lbl = os.path.join(output_dir, 'labels', split, f'{basename}.txt')
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(labels))

            print(f"OK ({len(labels)} annotations)")
        except Exception as e:
            print(f"FAILED: {e}")

    # Write data.yaml
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\nDataset written to {output_dir}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Classes: {CLASS_NAMES}")


if __name__ == '__main__':
    main()
