# AI Image Detection using MobileNet SSD

## 📌 Project Overview

This project is an AI-based Image Detection System developed using Python, OpenCV, and the MobileNet SSD deep learning model.

The system automatically detects and labels objects present in images stored in the input folder and saves the processed images with bounding boxes in the output folder.

---

## 🎯 Features

- Detects multiple objects in images
- Uses pre-trained MobileNet SSD model
- Draws bounding boxes around detected objects
- Displays confidence scores
- Processes multiple images automatically
- Saves results in an output folder

---

## 🧠 Technologies Used

- Python 3.11
- OpenCV
- NumPy
- MobileNet SSD
- Deep Neural Networks (DNN)

---

## 📂 Project Structure

```text
AI_IMAGE_DETECTION/
│
├── Inputs/
│   ├── image1.jpg
│   ├── image2.jpg
│
├── Outputs/
│   ├── detected_image1.jpg
│   ├── detected_image2.jpg
│
├── Models/
│   ├── MobileNetSSD_deploy.prototxt
│   ├── MobileNetSSD_deploy.caffemodel
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone <your-repository-link>
cd AI_IMAGE_DETECTION
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Place images inside the **Inputs** folder.

Run:

```bash
python app.py
```

Detected images will be saved automatically in the **Outputs** folder.

---

## 🔍 Detectable Objects

MobileNet SSD can detect:

- Person
- Car
- Bus
- Bicycle
- Dog
- Cat
- Chair
- Bottle
- Dining Table
- Sofa
- Train
- TV Monitor
- Horse
- Bird
- Sheep
- Cow
- Motorbike
- Aeroplane
- Boat
- Potted Plant

---

## 📸 Sample Output

The system generates output images with:

- Bounding Boxes
- Object Labels
- Confidence Scores

Example:

```text
Person: 0.98
Car: 0.95
Bus: 0.99
```

---

## 🚀 Future Improvements

- Real-time webcam detection
- Video object detection
- Web-based interface using Flask
- Object counting system
- YOLO-based detection for higher accuracy

---

## 👩‍💻 Author

Laiba Rafique

AI Image Detection Projects