"""
Run predictions on 10 random images from the dataset using the trained Fall Army Worm detector
"""
from ultralytics import YOLO
import os
import random
from pathlib import Path

def main():
    # Load the best trained model
    model_path = 'runs/detect/fall_army_worm_optimized/weights/best.pt'
    model = YOLO(model_path)

    print("="*70)
    print("FALL ARMY WORM DETECTION - RANDOM IMAGE PREDICTIONS")
    print("="*70)
    print(f"Using model: {model_path}\n")

    # Get all validation images
    val_images_dir = 'yolo_dataset_fall_army_warm/images/val'
    all_images = list(Path(val_images_dir).glob('*.jpg')) + \
                 list(Path(val_images_dir).glob('*.jpeg')) + \
                 list(Path(val_images_dir).glob('*.png'))

    # If we have fewer than 10 images in val, also include training images
    if len(all_images) < 10:
        train_images_dir = 'yolo_dataset_fall_army_warm/images/train'
        train_images = list(Path(train_images_dir).glob('*.jpg')) + \
                       list(Path(train_images_dir).glob('*.jpeg')) + \
                       list(Path(train_images_dir).glob('*.png'))
        all_images.extend(train_images)

    # Select 10 random images
    random.seed(42)  # For reproducibility
    selected_images = random.sample(all_images, min(10, len(all_images)))

    print(f"Total available images: {len(all_images)}")
    print(f"Selected {len(selected_images)} random images for prediction\n")
    print("="*70)

    # Run predictions
    results = model.predict(
        source=selected_images,
        conf=0.25,  # Confidence threshold
        iou=0.45,   # IoU threshold for NMS
        save=True,  # Save images with predictions
        save_txt=True,  # Save prediction results as txt
        save_conf=True,  # Save confidence scores
        project='runs/predict',
        name='fall_army_worm_random',
        exist_ok=True,
        line_width=2,
        show_labels=True,
        show_conf=True
    )

    print("\nPREDICTION RESULTS:")
    print("="*70)

    total_detections = 0
    for i, (img_path, result) in enumerate(zip(selected_images, results), 1):
        num_detections = len(result.boxes)
        total_detections += num_detections

        print(f"\n{i}. {img_path.name}")
        print(f"   Detections: {num_detections}")

        if num_detections > 0:
            for j, box in enumerate(result.boxes):
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                print(f"   - Detection {j+1}: fall_army_worm (confidence: {conf:.2%})")
                print(f"     Box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        else:
            print(f"   - No fall army worms detected")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images processed: {len(selected_images)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections/len(selected_images):.2f}")
    print(f"\nPrediction images saved to: runs/predict/fall_army_worm_random/")
    print("="*70)

if __name__ == "__main__":
    main()
