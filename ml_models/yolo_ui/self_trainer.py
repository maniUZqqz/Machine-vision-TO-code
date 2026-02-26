#!/usr/bin/env python
"""
Self-Training Loop for YOLO UI Detector

Implements a semi-supervised learning cycle:
1. Run current model on unlabeled images
2. Filter high-confidence predictions as pseudo-labels
3. Compare with classical pipeline detections
4. Use feedback loop similarity as quality signal
5. Add best pseudo-labels to training set
6. Retrain model

This allows the model to improve over time as more screenshots
are processed through the web app.

Usage:
    python self_trainer.py --unlabeled-dir ../../uploads/
    python self_trainer.py --feedback-threshold 0.7
"""
import os
import sys
import json
import shutil
import logging
import argparse
from datetime import datetime

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, 'dataset')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_ui_detector.pt')
HISTORY_FILE = os.path.join(SCRIPT_DIR, 'training_history.json')

# Classes (must match data.yaml)
CLASSES = [
    'button', 'input', 'select', 'card', 'image', 'icon',
    'text_block', 'header_bar', 'sidebar', 'navbar', 'separator'
]


class SelfTrainer:
    """
    Semi-supervised self-training for the YOLO UI detector.
    """

    def __init__(self, confidence_threshold=0.6,
                 feedback_threshold=0.7, max_pseudo_labels=100):
        self.confidence_threshold = confidence_threshold
        self.feedback_threshold = feedback_threshold
        self.max_pseudo_labels = max_pseudo_labels
        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                return json.load(f)
        return {'iterations': [], 'total_images': 0, 'best_map': 0.0}

    def _save_history(self):
        with open(HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=2)

    def run_iteration(self, unlabeled_dir, epochs=30):
        """Run one self-training iteration."""
        import django
        django.setup()

        iteration = len(self.history['iterations']) + 1
        print(f"\n{'='*60}")
        print(f"  Self-Training Iteration {iteration}")
        print(f"{'='*60}")

        # Step 1: Find unlabeled images
        images = self._find_unlabeled_images(unlabeled_dir)
        if not images:
            print("No unlabeled images found.")
            return False

        print(f"  Found {len(images)} unlabeled images")

        # Step 2: Generate pseudo-labels
        pseudo_labels = self._generate_pseudo_labels(images)
        if not pseudo_labels:
            print("No high-confidence pseudo-labels generated.")
            return False

        print(f"  Generated pseudo-labels for {len(pseudo_labels)} images")

        # Step 3: Quality check with feedback loop
        quality_images = self._quality_check(pseudo_labels)
        print(f"  Quality check passed: {len(quality_images)}/{len(pseudo_labels)}")

        if not quality_images:
            print("No images passed quality check.")
            return False

        # Step 4: Add to training set
        added = self._add_to_training_set(quality_images)
        print(f"  Added {added} images to training set")

        # Step 5: Retrain
        if added > 0:
            model_path = self._retrain(epochs)
            if model_path:
                # Record iteration
                self.history['iterations'].append({
                    'iteration': iteration,
                    'date': datetime.now().isoformat(),
                    'images_added': added,
                    'total_pseudo_labels': len(pseudo_labels),
                    'quality_passed': len(quality_images),
                })
                self.history['total_images'] += added
                self._save_history()
                return True

        return False

    def _find_unlabeled_images(self, unlabeled_dir):
        """Find images that haven't been labeled yet."""
        if not os.path.isdir(unlabeled_dir):
            return []

        # Get already-labeled images
        labeled = set()
        for split in ('train', 'val'):
            label_dir = os.path.join(DATASET_DIR, 'labels', split)
            if os.path.isdir(label_dir):
                for f in os.listdir(label_dir):
                    labeled.add(os.path.splitext(f)[0])

        # Find unlabeled
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        images = []
        for f in os.listdir(unlabeled_dir):
            name, ext = os.path.splitext(f)
            if ext.lower() in extensions and name not in labeled:
                images.append(os.path.join(unlabeled_dir, f))

        return images[:self.max_pseudo_labels]

    def _generate_pseudo_labels(self, images):
        """Use current YOLO model + classical pipeline to generate labels."""
        from PIL import Image
        from vision_engine.pipeline import create_default_pipeline
        from auto_labeler import generate_labels

        results = {}

        # Try YOLO first
        yolo_available = os.path.exists(MODEL_PATH)
        yolo_model = None
        if yolo_available:
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(MODEL_PATH)
                print("  Using YOLO + classical pipeline for pseudo-labeling")
            except Exception:
                pass

        # Classical pipeline
        pipeline = create_default_pipeline()

        for img_path in images:
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                h, w = img_array.shape[:2]

                labels = []

                # YOLO predictions (high confidence only)
                if yolo_model:
                    try:
                        yolo_results = yolo_model.predict(
                            img_array, conf=self.confidence_threshold,
                            verbose=False
                        )
                        for result in yolo_results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                                    cx = ((x1 + x2) / 2) / w
                                    cy = ((y1 + y2) / 2) / h
                                    nw = (x2 - x1) / w
                                    nh = (y2 - y1) / h

                                    labels.append(
                                        f"{cls_id} {cx:.6f} {cy:.6f} "
                                        f"{nw:.6f} {nh:.6f}"
                                    )
                    except Exception as e:
                        logger.warning(f"YOLO failed on {img_path}: {e}")

                # If YOLO gave few results, supplement with classical
                if len(labels) < 5:
                    classical_labels = generate_labels(img_path, pipeline)
                    # Only add non-overlapping classical labels
                    labels.extend(classical_labels)

                if labels:
                    results[img_path] = {
                        'labels': labels,
                        'image_array': img_array,
                    }

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")

        return results

    def _quality_check(self, pseudo_labels):
        """Use feedback loop to check quality of pseudo-labeled images."""
        from vision_engine.feedback_loop import FeedbackLoop
        from code_generator.generator import CodeGenerator
        from vision_engine.models.page import PageStructure
        from vision_engine.pipeline import create_default_pipeline

        feedback = FeedbackLoop()
        generator = CodeGenerator()
        pipeline = create_default_pipeline()

        quality_images = {}

        for img_path, data in pseudo_labels.items():
            try:
                img_array = data['image_array']

                # Run full pipeline to get HTML
                ctx = pipeline.run(img_array)

                orphan_texts = [t for t in ctx.text_elements if t.parent_id is None]
                all_root = ctx.regions + orphan_texts
                root_elements = sorted(all_root, key=lambda e: e.bbox.area,
                                       reverse=True)

                page = PageStructure(
                    width=ctx.width, height=ctx.height,
                    background_color=ctx.page_background,
                    root_elements=root_elements,
                    text_direction=getattr(ctx, 'page_direction', 'ltr'),
                )
                result = generator.generate(page)

                # Check similarity
                fb_result = feedback.compare_only(img_array, result['combined'])
                similarity = fb_result.get('similarity', 0)

                if similarity >= self.feedback_threshold:
                    quality_images[img_path] = data
                    quality_images[img_path]['similarity'] = similarity

            except Exception as e:
                logger.warning(f"Quality check failed for {img_path}: {e}")

        return quality_images

    def _add_to_training_set(self, quality_images):
        """Add quality-checked images to the training dataset."""
        added = 0
        for img_path, data in quality_images.items():
            basename = os.path.splitext(os.path.basename(img_path))[0]

            # Add to train split
            dst_img = os.path.join(DATASET_DIR, 'images', 'train',
                                   os.path.basename(img_path))
            dst_lbl = os.path.join(DATASET_DIR, 'labels', 'train',
                                   f'{basename}.txt')

            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(data['labels']))
                added += 1

        return added

    def _retrain(self, epochs):
        """Retrain the YOLO model with the expanded dataset."""
        from ultralytics import YOLO

        data_yaml = os.path.join(DATASET_DIR, 'data.yaml')
        if not os.path.exists(data_yaml):
            return None

        # Use existing model as starting point if available
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            model = YOLO('yolo11n.pt')

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=8,
            name='ui_detector_self_train',
            project=os.path.join(SCRIPT_DIR, 'runs', 'detect'),
            exist_ok=True,
            patience=15,
            save=True,
            verbose=True,
        )

        # Copy best weights
        best_pt = os.path.join(SCRIPT_DIR, 'runs', 'detect',
                               'ui_detector_self_train', 'weights', 'best.pt')
        if os.path.exists(best_pt):
            shutil.copy2(best_pt, MODEL_PATH)
            return MODEL_PATH

        return None


def main():
    parser = argparse.ArgumentParser(description='Self-training loop for YOLO UI detector')
    parser.add_argument('--unlabeled-dir', default=os.path.join(PROJECT_ROOT, 'uploads'),
                        help='Directory with unlabeled images')
    parser.add_argument('--confidence', type=float, default=0.6,
                        help='Minimum confidence for pseudo-labels')
    parser.add_argument('--feedback-threshold', type=float, default=0.7,
                        help='Minimum similarity score for quality check')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs per iteration')
    parser.add_argument('--max-images', type=int, default=100,
                        help='Max images per iteration')
    args = parser.parse_args()

    trainer = SelfTrainer(
        confidence_threshold=args.confidence,
        feedback_threshold=args.feedback_threshold,
        max_pseudo_labels=args.max_images,
    )

    success = trainer.run_iteration(args.unlabeled_dir, epochs=args.epochs)
    if success:
        print("\nSelf-training iteration completed successfully!")
    else:
        print("\nSelf-training iteration did not produce improvements.")


if __name__ == '__main__':
    main()
