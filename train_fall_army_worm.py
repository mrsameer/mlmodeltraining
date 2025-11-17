"""
Training script for Fall Army Worm detection using YOLOv8
"""
from ultralytics import YOLO
import torch

def main():
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano model for faster training

    # Train the model
    results = model.train(
        data='yolo_dataset_fall_army_warm/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='fall_army_worm_detector',
        patience=20,  # Early stopping patience
        save=True,
        device=device,
        workers=8,
        project='runs/detect',
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    # Validate the model
    print("\nValidating the model...")
    metrics = model.val()

    # Print results
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best model saved at: runs/detect/fall_army_worm_detector/weights/best.pt")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return model, results

if __name__ == "__main__":
    main()
