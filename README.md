# **Vehicle and License Plate Detection**

This project uses a YOLOv8 object detection model to detect vehicles and their license plates from images, followed by Optical Character Recognition (OCR) to extract the license plate numbers. This can be applied to scenarios such as automated toll systems, parking management, and law enforcement for vehicle identification.

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Model Training](#model-training)
5. [Inference and OCR](#inference-and-ocr)
6. [Results and Evaluation](#results-and-evaluation)
7. [Challenges](#challenges)
8. [Future Improvements](#future-improvements)

---

## **Project Overview**

This project involves two main tasks:
1. **Vehicle Detection**: Detecting vehicles and locating license plates in images using the YOLOv8 model.
2. **License Plate OCR**: Using Tesseract OCR to extract text from detected license plates.

We begin by annotating images of cars with bounding boxes around the license plates, training the YOLO model for detection, and then applying OCR on the detected plates to extract readable text.

---

## **Dataset**

The dataset includes:
- **Images**: Pictures of vehicles with visible license plates.
- **Annotations**: XML files containing bounding box information of the license plates.

We split the dataset into training, validation, and test sets to ensure proper model evaluation.

---

## **Installation**

To set up the environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/subu0106/Vehicle-Detection-and-License-Plate-Recognition.git
   cd Vehicle-Detection-and-License-Plate-Recognition
   ```

2. Install the required packages:
   ```bash
   pip install ultralytics opencv-python-headless numpy pytesseract pandas matplotlib scikit-learn wandb
   sudo apt-get install -y tesseract-ocr libtesseract-dev
   ```

3. Set up your environment:
   - For GPU users, ensure that your Google Colab or local environment has GPU enabled. Colab provides a free GPU option, but it comes with limitations on usage.
   
4. Set up your Tesseract OCR path:
   ```python
   pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
   ```

---

## **Model Training**

### **1. Data Preprocessing**

We convert XML annotations into YOLO-compatible format and split the dataset into training, validation, and test sets.

- **Function to convert annotations**: We created a function to extract bounding box information and convert it into the YOLO format.
- **Train/Test Split**: We used 10% of the data for testing and split the remaining into training and validation sets.

### **2. Training the YOLOv8 Model**

- The YOLOv8 model is trained using a `datasets.yaml` configuration file.
- The training process involves:
  - Setting hyperparameters (epochs, batch size, image size).
  - Monitoring the model's mAP (mean Average Precision) across different epochs.
  - Saving the trained model for further inference.

```python
# Train the YOLOv8 model
model.train(
    data="datasets.yaml",
    epochs=50,
    batch=16,
    device="cpu"  # Use 'cuda' for GPU
)
```

---

## **Inference and OCR**

### **Vehicle and License Plate Detection**

We use the trained YOLO model to detect vehicles and license plates. The bounding boxes are drawn on the detected plates, and the confidence scores are shown.

```python
def get_vehicle_image(image_name):
    results = model(image_path)
    # Plot the image with bounding boxes
```

### **License Plate OCR**

Once the license plate is detected, Tesseract OCR is used to extract the license plate number from the detected bounding box area.

```python
def get_numberplate_image(image_name):
    text = pytesseract.image_to_string(cropped_image)
    print(f"Detected text: {text}")
```

---

## **Results and Evaluation**

During training, the model’s performance was evaluated using metrics like **mAP@0.5** and **mAP@0.5:0.95**. These metrics show how well the model detects objects at different Intersection over Union (IoU) thresholds.

- **mAP (Mean Average Precision)**: This indicates the precision and recall balance of the model.
- **Plotting Accuracy**: Accuracy over different epochs was visualized to track the performance.

```python
# Plotting mAP over epochs
plt.plot(epochs, mAP_0_5, label="mAP@0.5")
plt.plot(epochs, mAP_0_5_0_95, label="mAP@0.5-0.95")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
```

---

## **Challenges**

1. **GPU Limitations**: During the development, we encountered GPU resource limits in Google Colab, forcing us to switch to CPU, which slowed down the training process significantly.
   
2. **OCR Accuracy**: The Tesseract OCR sometimes struggles with low-resolution images or distorted license plates, leading to imperfect text extraction.

---

## **Future Improvements**

1. **Use Cloud GPUs**: To overcome GPU limitations, using services like AWS, Azure, or GCP for faster and more scalable training.
   
2. **Data Augmentation**: Improve the model by adding more data augmentation techniques to make it more robust to varying conditions (e.g., different lighting, angles).
   
3. **OCR Enhancements**: Integrate more advanced OCR techniques or preprocessing methods like deblurring or enhancing the license plate area before OCR.

---

## **Contributors**

- [Subavarshana A](https://github.com/subu0106)

