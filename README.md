# 🔍 Anomaly Activity Detection Using Video Surveillance

An end-to-end **Deep Learning–based Suspicious Activity Detection System** designed to automate surveillance monitoring and enable **proactive real-time alerts** from CCTV video streams.

This project uses a **Convolutional Recurrent Neural Network (CRNN)** architecture to detect anomalous human activities such as **Assault, Arson, Fighting**, and other criminal behaviors from untrimmed real-world surveillance footage.

---

## 🚀 Motivation

Modern surveillance systems generate massive volumes of video data, making manual monitoring inefficient and error-prone due to human fatigue. This project addresses that bottleneck by transforming surveillance from a **reactive forensic tool** into a **proactive intelligent security system**.

---

## 🧠 System Overview

The proposed pipeline models video as a **spatiotemporal problem**:

1. **Spatial Feature Extraction**
   - ResNet-50 (pre-trained on ImageNet)
   - Extracts high-level visual features from individual frames

2. **Temporal Modeling**
   - LSTM network
   - Learns motion patterns and action evolution across frame sequences

3. **Decision Layer**
   - Fully connected classifier for anomaly prediction

📌 This hybrid design balances **accuracy and computational efficiency**, avoiding expensive 3D convolutions.

---

## 🏗 Architecture

![alt text](image.png)

**Pipeline:**  
`Input Frames → ResNet-50 → Feature Vectors → LSTM → Anomaly Classification`

---

## 📂 Dataset

- **UCF-Crime Dataset**
  - Real-world, untrimmed CCTV footage
  - 14 activity classes including Normal, Assault, Arson, Fighting, Robbery, etc.
  - Highly imbalanced and temporally sparse anomalies

![alt text](image-1.png)

---

## ⚙ Methodology

- **Temporal Subsampling**
  - Divides long videos into fixed segments
  - Extracts representative frames to reduce redundancy

- **Transfer Learning**
  - Frozen CNN backbone to leverage pre-trained visual features

- **Sequence Modeling**
  - Frame features grouped into clips and passed to LSTM

---

## 📊 Results

- Successfully detects complex anomalies from background activity
- Demonstrates stable convergence and reliable classification
- Suitable foundation for scalable intelligent surveillance systems

### Sample Output

![alt text](image-2.png)
![alt text](image-3.png)

---

## 🧪 Evaluation Metrics

- Confusion Matrix
- Precision, Recall, F1-score
- Validation Loss Monitoring


---

## 🛠 Tech Stack

- **Languages:** Python 3.8+
- **Frameworks:** PyTorch / TensorFlow
- **Vision:** OpenCV, Pillow
- **Models:** ResNet-50, LSTM
- **Hardware:** NVIDIA GPU recommended (CUDA-enabled)

---

## 📌 Key Features

- Real-world anomaly detection (not staged data)
- Efficient CRNN-based architecture
- Scalable for long, untrimmed videos
- Designed for real-time surveillance adaptation

---

## 🔮 Future Scope

- Extended training with hyperparameter tuning
- Data augmentation for robustness
- Model quantization for edge deployment
- Real-time alerting and deployment integration

---

## 👨‍💻 Authors

- **More Kshitij**
- **Patadiya Vishesh**
- **Tanpure Anushka**
- **Waghchaure Saraswati**

**Guided by:** Dr. S. S. Mohite  
COEP Technological University

---

## 📜 License

This project is intended for academic and research purposes.
