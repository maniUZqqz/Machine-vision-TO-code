"""
Django management command to train/retrain the YOLO UI detector.

Usage:
    python manage.py train_yolo                # Full training
    python manage.py train_yolo --self-train   # Self-training iteration
    python manage.py train_yolo --data-only    # Only generate data
"""
import os
import sys

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Train or retrain the YOLO UI element detector'

    def add_arguments(self, parser):
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--batch', type=int, default=8)
        parser.add_argument('--synthetic-count', type=int, default=500)
        parser.add_argument('--self-train', action='store_true',
                            help='Run self-training iteration')
        parser.add_argument('--data-only', action='store_true',
                            help='Only generate data')
        parser.add_argument('--skip-synthetic', action='store_true')

    def handle(self, *args, **options):
        ml_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..',
            'ml_models', 'yolo_ui'
        )
        sys.path.insert(0, os.path.abspath(ml_dir))

        if options['self_train']:
            self.stdout.write('Running self-training iteration...')
            from self_trainer import SelfTrainer

            # Use uploaded images directory
            uploads_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', '..', '..',
                'media', 'uploads'
            )
            trainer = SelfTrainer()
            success = trainer.run_iteration(
                uploads_dir, epochs=options['epochs']
            )
            if success:
                self.stdout.write(self.style.SUCCESS(
                    'Self-training completed successfully!'
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    'Self-training did not produce improvements.'
                ))
        else:
            self.stdout.write('Running YOLO training...')
            from train import generate_data, train_model

            class Args:
                pass

            train_args = Args()
            train_args.skip_synthetic = options['skip_synthetic']
            train_args.synthetic_count = options['synthetic_count']
            train_args.epochs = options['epochs']
            train_args.batch = options['batch']
            train_args.imgsz = 640
            train_args.model_size = 'n'
            train_args.resume = False
            train_args.data_only = options['data_only']

            dataset_dir = generate_data(train_args)

            if not options['data_only']:
                model_path = train_model(dataset_dir, train_args)
                if model_path:
                    self.stdout.write(self.style.SUCCESS(
                        f'Training complete! Model: {model_path}'
                    ))
                else:
                    self.stdout.write(self.style.ERROR('Training failed.'))
