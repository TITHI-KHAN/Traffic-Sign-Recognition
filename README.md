# Traffic-Sign-Recognition

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/0a3845d9-dfc7-4c65-9ee7-1b94bb38a1ac)

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/e5944e4d-a622-4ce0-a574-777bf595d6ee)

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/ee0d6214-1d66-413e-b0f3-a08a28423b29)

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/d8552b0b-668a-4265-bf7c-980a4ad77372)

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/2efde7e9-78d4-4e0b-9770-e8bfb8adfb0c)

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/fcd245db-2197-4a52-92e0-ec0c43683406)

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/7dd651dd-dec9-43a2-b39d-007a45bc0752)

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/a7a15bdb-77c1-4e76-b70a-7e15ac1a6790)

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/68f4c7fa-750a-4099-833c-2327c566ade3)

![image](https://github.com/TITHI-KHAN/Traffic-Sign-Recognition/assets/65033964/e26d4faa-2589-4251-b8a1-6105fc3f32a4)

## Dataset

Traffic : https://www.kaggle.com/datasets/tithikhan/traffic

## Step-by-Step Explanation

This code is a comprehensive set of functions and scripts for performing various image processing and computer vision tasks, particularly focused on traffic sign recognition. 

Let's break down the code into sections and provide a brief explanation of each part.

### 1. Importing Libraries and Setting Directories
```python
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
- This section imports necessary libraries for image processing, file system operations, data manipulation, and visualization.

### 2. Defining Directory and File Operations
```python
dir = "/kaggle/input/traffic/myData"
extracted_files = os.listdir(dir)
```
- Sets the directory path where the image data is stored and lists all files in that directory.

### 3. Displaying Sample Images
```python
def display_sample_images(base_path, sample_files, num_samples=4):
    ...
```
- Function to display a few sample images from the dataset.

### 4. Displaying Sample Images Using OpenCV
```python
def display_sample_images_cv(base_path, num_samples=4):
    ...
```
- Similar to the previous function but displays images using OpenCV instead of Matplotlib.

### 5. Displaying Image Histograms
```python
def display_image_histograms(base_path, num_samples=2):
    ...
```
- Displays histograms of pixel intensities for sample images.

### 6. Edge Detection
```python
def display_edge_detection(base_path, num_samples=2):
    ...
```
- Detects edges in sample images using the Canny edge detector.

### 7. Thresholding Images
```python
def display_threshold_images(base_path, num_samples=2):
    ...
```
- Applies adaptive thresholding to sample images.

### 8. Sobel Edge Detection
```python
def display_sobel_edge_images(base_path, num_samples=2):
    ...
```
- Performs Sobel edge detection on sample images.

### 9. Optical Flow
```python
def display_optical_flow_images_corrected(base_path, num_samples=2):
    ...
```
- Computes optical flow between consecutive frames in sample images.

### 10. Watershed Segmentation
```python
def display_watershed_segmentation(base_path, num_samples=2):
    ...
```
- Applies watershed segmentation to sample images.

### 11. Image Inpainting
```python
def display_image_inpainting(base_path, num_samples=2):
    ...
```
- Performs image inpainting on sample images.

### 12. Thermal Effect
```python
def display_thermal_effect(base_path, num_samples=2):
    ...
```
- Applies a thermal effect to sample images.

### 13. Edge Collage
```python
def display_edge_collage(base_path, num_samples=4):
    ...
```
- Creates a collage of images showing various edge detection methods.

### 14. Custom Convolution
```python
def display_custom_convolution(base_path, num_samples=4):
    ...
```
- Applies custom convolution kernels to sample images.

### 15. Creating Label Images
```python
labels_data = pd.read_csv('/kaggle/input/traffic/labels.csv')
```
- Reads label data from a CSV file containing traffic sign labels.

### 16. Loading Image Data
```python
def load_data(data_directory, target_size=(64, 64)):
    ...
```
- Function to load image data from directories, preprocess them, and split into training and test sets.

### 17. Creating a CNN Model
```python
def create_model(input_shape, num_classes):
    ...
```
- Defines a convolutional neural network (CNN) model architecture for traffic sign classification.

### 18. Compiling and Training the Model
```python
input_shape = (64, 64, 3)
num_classes = 43
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)
```
- Compiles the model, specifying optimizer, loss function, and evaluation metrics, then trains the model on the training data.

### 19. Evaluating Model Performance
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.4f}")
```
- Evaluates the trained model on the test dataset and prints the test accuracy.

### 20. Plotting Training History
```python
if 'history' in globals():
    ...
```
- Plots the training and validation accuracy and loss over epochs.

### 21. Plotting Predictions
```python
def plot_images(images, actual_labels, predicted_labels, class_names):
    ...
```
- Function to plot sample images with their predicted and actual labels.

### 22. Fine-Tuning a Pretrained Model
```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
...
```
- Loads a pretrained MobileNetV2 model, freezes its layers, and adds custom classification layers on top.

### 23. Compiling and Training the Fine-Tuned Model
```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
```
- Compiles and trains the fine-tuned model.

### 24. Plotting Predictions for Fine-Tuned Model
```python
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(test_labels, axis=1) if test_labels.ndim > 1 else test_labels
plot_images(test_images, actual_labels, predicted_labels, class_names)
```
- Plots predictions for the fine-tuned model.

This code provides a comprehensive workflow for image processing, model training, and evaluation, particularly focused on traffic sign recognition. Each section performs a specific task, such as data preprocessing, model definition, training, evaluation, and visualization.
