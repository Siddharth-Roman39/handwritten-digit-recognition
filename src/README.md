# 🖊️ Handwritten Digit Recognition (MNIST-based)

This project is a **Flask + TensorFlow web app** that can recognize handwritten digits (0–9) from either:
- A **canvas** where you draw digits  
- An **uploaded image** (scanned/phone photo of handwritten digit)

It uses a **CNN trained on the MNIST dataset** and processes input images into MNIST-style `28x28` format before prediction.

---

## 🚀 Features
- Draw digits on a canvas and get instant predictions.
- Upload handwritten digit images (e.g., from a notebook) and predict.
- Image preprocessing pipeline (contrast, noise reduction, binarization, centering).
- Pre-trained CNN model (trained on MNIST dataset).
- Flask backend with simple frontend UI.

---

## 🛠️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/digit-recognition.git
cd digit-recognition
