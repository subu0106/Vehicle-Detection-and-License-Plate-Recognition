#!/usr/bin/env python3
"""Train YOLOv8 license plate detection model."""

import os
import sys
from pathlib import Path
import shutil
import re
import pandas as pd
import xml.etree.ElementTree as xet
from glob import glob
from sklearn.model_selection import train_test_split
import cv2

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from src.utils.logger import setup_logger

logger = setup_logger('train_model')

def num_in_name(filename):
    """Extract number from filename for sorting."""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group(0))
    else:
        return 0

def load_dataset(dataset_path):
    """Load dataset from XML annotations."""
    logger.info(f"Loading dataset from {dataset_path}")

    # Creating a dictionary for storing Data
    data_dict = dict(
        img_path=[],
        xmin=[], ymin=[],
        xmax=[], ymax=[],
        img_w=[], img_h=[]
    )

    filepaths = glob(f"{dataset_path}/annotations/*.xml")
    logger.info(f"Found {len(filepaths)} annotation files")

    # Process each XML annotation file
    for filepath in sorted(filepaths, key=num_in_name):
        parser = xet.parse(filepath)
        root = parser.getroot()

        objectData = root.find("object")
        if objectData is None:
            continue

        bndbox = objectData.find("bndbox")
        if bndbox is None:
            continue

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Get the image filename and construct the full path to the image
        img_name = root.find('filename').text
        img_path = os.path.join(dataset_path, 'images', img_name)

        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue

        imgTensor = cv2.imread(img_path)
        if imgTensor is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        height, width, _ = imgTensor.shape

        # Save info into dict
        data_dict["img_path"].append(img_path)
        data_dict["xmin"].append(xmin)
        data_dict["ymin"].append(ymin)
        data_dict["xmax"].append(xmax)
        data_dict["ymax"].append(ymax)
        data_dict["img_w"].append(width)
        data_dict["img_h"].append(height)

    # Creating DataFrame
    df = pd.DataFrame(data_dict)
    logger.info(f"Loaded {len(df)} samples")
    return df

def create_train_test_split(df):
    """Split dataset into train, validation, and test sets."""
    logger.info("Splitting dataset into train/val/test")

    # 10% for test, rest for train
    train, test = train_test_split(df, test_size=0.1, random_state=42)

    # From train, take ~11% (1/9) for validation
    train, val = train_test_split(train, test_size=1/9, random_state=42)

    logger.info(f"Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
    return train, val, test

def create_yolo_folder(name, dataframe, output_dir="datasets"):
    """Create YOLO format folders and convert annotations."""
    logger.info(f"Creating YOLO format data for {name}")

    # Creating directory for labels and images
    labels_path = os.path.join(output_dir, name, "labels")
    image_path = os.path.join(output_dir, name, "images")

    # Create all folders
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)

    # Iterate through dataset and create YOLO format files
    for _, row in dataframe.iterrows():
        image_name, image_extension = os.path.splitext(os.path.basename(row["img_path"]))

        # Calculate YOLO format coordinates (normalized center x, y, width, height)
        x_center = (row["xmin"] + row["xmax"]) / 2 / row["img_w"]
        y_center = (row["ymin"] + row["ymax"]) / 2 / row["img_h"]
        width = (row['xmax'] - row['xmin']) / row['img_w']
        height = (row['ymax'] - row['ymin']) / row['img_h']

        # Save data in YOLO format (class_id x_center y_center width height)
        label_path = os.path.join(labels_path, f"{image_name}.txt")
        with open(label_path, "w") as file:
            file.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

        # Copy image to new directory
        shutil.copy(row["img_path"], os.path.join(image_path, image_name + image_extension))

    logger.info(f"Created {name} dataset at {image_path}")

def create_dataset_yaml(output_dir="datasets"):
    """Create dataset.yaml configuration file for YOLO training."""
    logger.info("Creating dataset.yaml configuration")

    # Get absolute path to the dataset directory
    abs_path = os.path.abspath(output_dir)

    dataset_yaml = f"""# Dataset configuration for YOLOv8 license plate detection

# Root directory (absolute path)
path: {abs_path}

# Directories (relative to path)
train: train/images
val: validation/images
test: test/images

# Number of classes
nc: 1

# Class names
names: ['license_plate']
"""

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as file:
        file.write(dataset_yaml)

    logger.info(f"Created dataset configuration at {yaml_path}")
    return yaml_path

def train_model(dataset_yaml_path, epochs=50, batch=16, imgsz=320, device='cpu'):
    """Train YOLOv8 model for license plate detection."""
    logger.info("Starting model training")
    logger.info(f"Parameters: epochs={epochs}, batch={batch}, imgsz={imgsz}, device={device}")

    # Load YOLOv8 nano model (pre-trained weights)
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        batch=batch,
        device=device,
        imgsz=imgsz,
        cache=True,  # Cache images for faster training
        project='runs/detect',
        name='train',
        exist_ok=True
    )

    logger.info("Training completed")
    return model, results

def save_best_model(source_path='runs/detect/train/weights/best.pt', target_path='models/best_license_plate_model.pt'):
    """Copy the best trained model to models directory."""
    logger.info(f"Saving best model from {source_path} to {target_path}")

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
        logger.info(f"Model saved successfully to {target_path}")
        return True
    else:
        logger.error(f"Source model not found: {source_path}")
        return False

def main():
    """Main training pipeline."""
    try:
        logger.info("="*70)
        logger.info("YOLO LICENSE PLATE DETECTION MODEL TRAINING")
        logger.info("="*70)

        # Configuration
        dataset_path = "data"
        output_dir = "datasets"
        epochs = 5  # Quick test with 5 epochs
        batch_size = 16
        image_size = 320
        device = 'cpu'  # Change to 'cuda' or '0' for GPU

        # Step 1: Load dataset
        logger.info("\n[1/6] Loading dataset...")
        df = load_dataset(dataset_path)

        if len(df) == 0:
            logger.error("No data found! Please ensure dataset is in the correct location.")
            sys.exit(1)

        # Step 2: Split dataset
        logger.info("\n[2/6] Splitting dataset...")
        train_df, val_df, test_df = create_train_test_split(df)

        # Step 3: Remove old datasets directory if it exists
        if os.path.exists(output_dir):
            logger.info(f"Removing old {output_dir} directory...")
            shutil.rmtree(output_dir)

        # Step 4: Create YOLO format datasets
        logger.info("\n[3/6] Creating YOLO format datasets...")
        create_yolo_folder("train", train_df, output_dir)
        create_yolo_folder("validation", val_df, output_dir)
        create_yolo_folder("test", test_df, output_dir)

        # Step 5: Create dataset configuration
        logger.info("\n[4/6] Creating dataset configuration...")
        yaml_path = create_dataset_yaml(output_dir)

        # Step 6: Train model
        logger.info("\n[5/6] Training model...")
        logger.info("This may take a while depending on your hardware...")
        model, results = train_model(yaml_path, epochs=epochs, batch=batch_size, imgsz=image_size, device=device)

        # Step 7: Save best model
        logger.info("\n[6/6] Saving best model...")
        if save_best_model():
            logger.info("\n" + "="*70)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info(f"Best model saved to: models/best_license_plate_model.pt")
            logger.info(f"Training results saved to: runs/detect/train/")
            logger.info("\nYou can now use the trained model with the web application!")
        else:
            logger.error("Failed to save the best model")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
