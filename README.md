# Suspicious Activity Detection using YOLO and Roboflow

## 1. Project Title

**Suspicious Activity Detection using YOLO and Roboflow**

---

# 2. Introduction

Suspicious activity detection is an important application of computer vision in modern surveillance systems. Monitoring CCTV cameras manually is difficult and time-consuming. Therefore, automated systems are required to detect suspicious activities such as:

* Weapon presence (gun, knife)
* Fighting between people
* Intrusion into restricted areas

This project aims to develop a **computer vision-based detection system** that can automatically identify suspicious activities in images and videos.

The system uses **YOLO (You Only Look Once)** for object detection and **Roboflow** for dataset preparation, annotation, and training.

The final goal of this project is to **deploy the detection system on embedded hardware (such as Raspberry Pi or Jetson Nano) for real-time surveillance monitoring**.

---

# 3. Objectives

The main objectives of this project are:

* To prepare a dataset for suspicious activity detection
* To annotate images using Roboflow
* To train a YOLO-based object detection model
* To detect suspicious objects or activities in images and videos
* To build a prototype for future hardware deployment

---

# 4. Tools and Technologies Used

| Tool              | Purpose                           |
| ----------------- | --------------------------------- |
| Roboflow          | Dataset management and annotation |
| YOLOv8            | Object detection model            |
| Python            | Model training and testing        |
| OpenCV            | Video frame extraction            |
| Google Colab      | Model testing and execution       |
| Roboflow Universe | Public dataset source             |

---

# 5. Project Workflow

The complete system workflow is shown below.

```
Video Input
     ↓
Frame Extraction
     ↓
Dataset Preparation
     ↓
Annotation (Roboflow)
     ↓
Dataset Generation
     ↓
Model Training (YOLO)
     ↓
Detection on Image / Video
```

---

# 6. Dataset Preparation

## 6.1 Video Dataset

Initially, a surveillance video containing suspicious activity was selected.

Since YOLO models require **images for training**, the video was converted into multiple image frames.

Approximately **60–70 frames** were extracted from the video to create the dataset.

---

## 6.2 Frame Extraction

Frames were extracted using Python and OpenCV.

Example code:

```python
import cv2

video = cv2.VideoCapture("video.mp4")
count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    cv2.imwrite(f"frame_{count}.jpg", frame)
    count += 1

video.release()
```

This process converts the video into multiple images which can be used for training.

---

# 7. Dataset Annotation using Roboflow

After extracting frames, the images were uploaded to Roboflow.

Steps followed:

1. Create a new project in Roboflow
2. Select **Object Detection** as the project type
3. Upload extracted image frames
4. Annotate objects using bounding boxes

### Labels used in the project

| Label  | Description       |
| ------ | ----------------- |
| person | human detection   |
| gun    | weapon detection  |
| knife  | weapon detection  |
| fight  | fighting activity |

Bounding boxes were drawn around objects or actions in the images.

---

# 8. Automatic Annotation

Roboflow provides an **Auto Labeling feature** which can automatically detect common objects like persons.

Workflow used:

```
Upload Images
     ↓
Auto Label Person
     ↓
Manual Correction
     ↓
Final Annotation
```

This reduced the annotation time significantly.

---

# 9. Dataset Generation

After annotation, a **dataset version** was generated.

Dataset preprocessing included:

* Image resizing to **640 × 640**
* Auto orientation correction
* Image normalization

---

# 10. Data Augmentation

To improve model performance, data augmentation techniques were applied:

* Image flipping
* Rotation
* Brightness adjustment
* Noise addition

This increased the effective dataset size and improved training.

---

# 11. Combining Multiple Datasets

To improve suspicious activity detection, multiple datasets were combined.

Datasets used:

1. Weapon Detection Dataset
2. Fight Detection Dataset

Steps followed:

1. Create a new project in Roboflow
2. Add weapon dataset
3. Add fight dataset to the same project
4. Ensure label consistency
5. Generate dataset version

This created a **multi-class dataset** containing:

```
person
gun
knife
fight
```

---

# 12. Model Training

The dataset was exported in **YOLO format** and trained using YOLOv8.

Example training code:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)
```

The training process allowed the model to learn patterns of suspicious activities.

---

# 13. Model Testing

The trained model was tested on images and videos.

### Image Detection

```python
model.predict("test.jpg", show=True)
```

The model detects objects and draws bounding boxes around them.

Example output:

```
person – 0.92
gun – 0.87
```

---

# 14. Video Detection

Although YOLO is trained on images, it can detect objects in videos by analyzing each frame.

Example code:

```python
model.predict(
    source="video.mp4",
    save=True
)
```

The system processes the video frame by frame and detects suspicious activities.

---

# 15. Testing in Google Colab

The model was tested using Google Colab.

Steps performed:

1. Install YOLO

```
pip install ultralytics
```

2. Upload trained model and video

3. Run detection

```python
from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(source="video.mp4", save=True)
```

The output video contains bounding boxes showing detected objects.

---

# 16. Results

The trained model was able to detect:

* Person in the scene
* Weapon objects such as gun or knife
* Fighting activity

Detection results were visualized using bounding boxes and confidence scores.

---

# 17. Limitations

Since the dataset used for training contained only **60–70 frames**, the model performance was limited.

Challenges observed:

* Incorrect detection in some frames
* Missed objects in low-quality images
* Lower confidence scores

These issues can be improved by increasing dataset size.

---

# 18. Future Scope

Future improvements include:

* Increasing dataset size
* Training with larger datasets
* Improving detection accuracy
* Deploying the model on embedded hardware
* Real-time CCTV monitoring

---

# 19. Hardware Implementation Plan

The final system will work as follows:

```
CCTV Camera
     ↓
Embedded Device (Raspberry Pi / Jetson Nano)
     ↓
YOLO Detection Model
     ↓
Suspicious Activity Alert
```

This will allow real-time detection of suspicious activities.

---

# 20. Conclusion

This project demonstrates the use of computer vision for suspicious activity detection. Using Roboflow for dataset preparation and YOLO for object detection, a prototype model was developed to detect activities such as weapon presence and fighting.

Although the current model uses a small dataset, the project successfully demonstrates the pipeline required for building an automated surveillance detection system. Future work will focus on improving the dataset and deploying the model on hardware for real-time monitoring.
