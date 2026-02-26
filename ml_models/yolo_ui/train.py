#!/usr/bin/env python
"""
YOLO UI Element Detection — Training Script

Trains a YOLOv8 model to detect UI elements in screenshots.

Steps:
  1. Generate synthetic training data (if dataset doesn't exist)
  2. Auto-label any real images from test_samples/
  3. Train YOLOv8 model
  4. Export to ONNX for inference

Usage:
    python train.py                    # Full pipeline
    python train.py --skip-synthetic   # Skip synthetic data generation
    python train.py --epochs 100       # Train for 100 epochs
    python train.py --resume           # Resume training from last checkpoint
"""
import os
import sys
import argparse
import shutil

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)


def generate_data(args):
    """Generate training data: synthetic + auto-labeled real images."""
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')

    # Step 1: Generate synthetic images
    if not args.skip_synthetic:
        print("=" * 60)
        print("  Step 1: Generating synthetic training data")
        print("=" * 60)
        from synthetic_generator import SyntheticUIGenerator
        gen = SyntheticUIGenerator()
        gen.generate_batch(args.synthetic_count, dataset_dir, split_ratio=0.85)
        print()

    # Step 2: Auto-label real images
    real_images_dir = os.path.join(PROJECT_ROOT, 'test_samples', 'input')
    if os.path.isdir(real_images_dir):
        real_images = [
            f for f in os.listdir(real_images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if real_images:
            print("=" * 60)
            print(f"  Step 2: Auto-labeling {len(real_images)} real images")
            print("=" * 60)

            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
            import django
            django.setup()

            from auto_labeler import generate_labels
            from vision_engine.pipeline import create_default_pipeline
            pipeline = create_default_pipeline()

            # Put real images in both train and val for max coverage
            for split in ('train', 'val'):
                for img_file in real_images:
                    src = os.path.join(real_images_dir, img_file)
                    dst_img = os.path.join(dataset_dir, 'images', split, img_file)
                    dst_lbl = os.path.join(dataset_dir, 'labels', split,
                                           os.path.splitext(img_file)[0] + '.txt')

                    if not os.path.exists(dst_img):
                        shutil.copy2(src, dst_img)

                    if not os.path.exists(dst_lbl):
                        try:
                            labels = generate_labels(src, pipeline)
                            with open(dst_lbl, 'w') as f:
                                f.write('\n'.join(labels))
                            print(f"  Labeled: {img_file} ({len(labels)} annotations)")
                        except Exception as e:
                            print(f"  Failed: {img_file}: {e}")
            print()

    return dataset_dir


def train_model(dataset_dir, args):
    """Train YOLOv8 model."""
    from ultralytics import YOLO

    print("=" * 60)
    print("  Step 3: Training YOLO model")
    print("=" * 60)

    data_yaml = os.path.join(dataset_dir, 'data.yaml')
    if not os.path.exists(data_yaml):
        print(f"ERROR: data.yaml not found at {data_yaml}")
        print("Run data generation first.")
        return None

    # Use YOLOv8n (nano) for fast training, or YOLOv8s for better accuracy
    model_size = args.model_size  # 'n', 's', 'm'
    base_model = f'yolo11{model_size}.pt'

    if args.resume:
        # Resume from last checkpoint
        last_pt = os.path.join(os.path.dirname(__file__), 'runs', 'detect',
                               'ui_detector', 'weights', 'last.pt')
        if os.path.exists(last_pt):
            print(f"Resuming from {last_pt}")
            model = YOLO(last_pt)
        else:
            print(f"No checkpoint found, starting fresh with {base_model}")
            model = YOLO(base_model)
    else:
        model = YOLO(base_model)

    # Training parameters
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name='ui_detector',
        project=os.path.join(os.path.dirname(__file__), 'runs', 'detect'),
        exist_ok=True,
        patience=20,        # Early stopping
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=0,          # No rotation for UI
        translate=0.1,
        scale=0.3,
        flipud=0.0,         # No vertical flip for UI
        fliplr=0.0,         # No horizontal flip (UI is directional)
        mosaic=0.5,
        mixup=0.1,
    )

    # Copy best weights to a known location
    best_pt = os.path.join(os.path.dirname(__file__), 'runs', 'detect',
                           'ui_detector', 'weights', 'best.pt')
    final_pt = os.path.join(os.path.dirname(__file__), 'best_ui_detector.pt')

    if os.path.exists(best_pt):
        shutil.copy2(best_pt, final_pt)
        print(f"\nBest model saved to: {final_pt}")

    return final_pt


def export_model(model_path, args):
    """Export model to ONNX."""
    if not model_path or not os.path.exists(model_path):
        print("No model to export.")
        return

    from ultralytics import YOLO

    print("=" * 60)
    print("  Step 4: Exporting model")
    print("=" * 60)

    model = YOLO(model_path)

    # Export to ONNX
    if args.export_onnx:
        onnx_path = model.export(format='onnx', imgsz=args.imgsz)
        print(f"ONNX model: {onnx_path}")

    print("\nTraining complete!")
    print(f"Model: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train YOLO UI Element Detector')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--model-size', default='n', choices=['n', 's', 'm'],
                        help='YOLO model size: n(ano), s(mall), m(edium)')
    parser.add_argument('--synthetic-count', type=int, default=500,
                        help='Number of synthetic images to generate')
    parser.add_argument('--skip-synthetic', action='store_true',
                        help='Skip synthetic data generation')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export to ONNX after training')
    parser.add_argument('--data-only', action='store_true',
                        help='Only generate data, skip training')
    args = parser.parse_args()

    # Generate data
    dataset_dir = generate_data(args)

    if args.data_only:
        print("Data generation complete. Skipping training.")
        return

    # Train
    model_path = train_model(dataset_dir, args)

    # Export
    if model_path and args.export_onnx:
        export_model(model_path, args)


if __name__ == '__main__':
    main()
