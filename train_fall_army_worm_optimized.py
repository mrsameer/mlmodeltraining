"""
Optimized Training script for Fall Army Worm detection using YOLOv8
Based on research and best practices for small datasets
"""
from ultralytics import YOLO
import torch

def main():
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("OPTIMIZED YOLOV8 TRAINING FOR SMALL DATASET")
    print("="*70)
    print("\nDataset Statistics:")
    print("- Training images: 177")
    print("- Validation images: 15")
    print("- Total objects: ~282")
    print("- Avg objects/image: 1.59")
    print("\nKey Optimizations Applied:")
    print("1. Using YOLOv8s (small) instead of nano for better feature extraction")
    print("2. Reduced learning rate for stable convergence")
    print("3. Optimized augmentation for small dataset")
    print("4. Increased patience for early stopping")
    print("5. Adjusted loss weights for better recall")
    print("6. Close mosaic augmentation earlier to stabilize training")
    print("="*70 + "\n")

    # Load YOLOv8 Small model (better than nano for small datasets)
    model = YOLO('yolov8s.pt')

    # Optimized training parameters for small dataset
    results = model.train(
        # Data configuration
        data='yolo_dataset_fall_army_warm/data.yaml',

        # Training duration
        epochs=150,  # More epochs for small dataset
        patience=30,  # Increased patience for better convergence

        # Image and batch settings
        imgsz=640,  # Standard size good for pest detection
        batch=8,  # Smaller batch for better gradient updates on small dataset

        # Hardware
        device=device,
        workers=4,  # Reduced workers to prevent data loading issues

        # Output settings
        project='runs/detect',
        name='fall_army_worm_optimized',
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs

        # Learning rate (CRITICAL for small datasets)
        lr0=0.001,  # Lower initial learning rate for stability
        lrf=0.001,  # Lower final learning rate
        momentum=0.937,
        weight_decay=0.0005,

        # Loss function weights (optimized for better recall)
        box=7.5,  # Bounding box loss weight
        cls=0.3,  # Reduced classification loss for single class
        dfl=1.5,  # Distribution focal loss

        # Data augmentation (optimized for small dataset)
        hsv_h=0.01,  # Reduced hue augmentation
        hsv_s=0.5,  # Moderate saturation
        hsv_v=0.3,  # Moderate brightness
        degrees=15.0,  # Increased rotation for more variety
        translate=0.15,  # Increased translation
        scale=0.7,  # Increased scale variation
        shear=2.0,  # Added slight shear
        perspective=0.0001,  # Minimal perspective
        flipud=0.0,  # No vertical flip (pests are usually upright)
        fliplr=0.5,  # 50% horizontal flip

        # Advanced augmentation
        mosaic=1.0,  # Use mosaic augmentation
        mixup=0.1,  # 10% mixup for regularization
        copy_paste=0.1,  # 10% copy-paste augmentation
        close_mosaic=100,  # Close mosaic at epoch 100 for fine-tuning

        # Regularization
        dropout=0.0,  # No dropout for small dataset

        # Performance optimization
        amp=True,  # Automatic mixed precision
        cache=False,  # Don't cache (small dataset)

        # Validation
        val=True,
        plots=True,  # Generate training plots

        # Other settings
        verbose=True,
        seed=42,  # For reproducibility
        deterministic=False,  # Allow for faster training
    )

    # Validate the model
    print("\n" + "="*70)
    print("FINAL VALIDATION")
    print("="*70)
    metrics = model.val()

    # Print comprehensive results
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest model saved at: runs/detect/fall_army_worm_optimized/weights/best.pt")
    print(f"Last model saved at: runs/detect/fall_army_worm_optimized/weights/last.pt")
    print("\nFinal Metrics:")
    print(f"  - Precision: {metrics.box.p:.4f}")
    print(f"  - Recall: {metrics.box.r:.4f}")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - F1 Score: {2 * (metrics.box.p * metrics.box.r) / (metrics.box.p + metrics.box.r + 1e-6):.4f}")
    print("\nRecommendations:")
    if metrics.box.r < 0.5:
        print("  - Recall is low. Consider:")
        print("    * Adding more training data")
        print("    * Adjusting confidence threshold during inference")
        print("    * Fine-tuning with lower cls loss weight")
    if metrics.box.p < 0.5:
        print("  - Precision is low. Consider:")
        print("    * Reviewing label quality")
        print("    * Increasing cls loss weight")
    if metrics.box.map50 > 0.5:
        print("  - Good performance! Model is ready for deployment.")
    print("="*70 + "\n")

    return model, results, metrics

if __name__ == "__main__":
    main()
