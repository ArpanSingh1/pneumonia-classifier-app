# Pneumonia Classifier 🫁

A CNN-based web app that predicts whether a chest X-ray shows signs of **pneumonia** or is **normal**, built with TensorFlow/Keras and deployed via Gradio on Render.

🔗 **Live App:** https://pneumonia-classifier-app-6evf.onrender.com/

---

## 📌 Overview

This project uses a Convolutional Neural Network (CNN) trained on chest X-ray images to classify them into two categories:
- **NORMAL**
- **PNEUMONIA**

The trained model is served through a simple Gradio interface, allowing users to upload an X-ray image and get an instant prediction with confidence scores.

---

## 🧠 Model Architecture

```
Conv2D(32, 3x3, relu) → MaxPooling2D(2x2)
Conv2D(32, 3x3, relu) → MaxPooling2D(2x2)
Flatten
Dense(128, relu)
Dense(1, sigmoid)
```

- **Input size:** 64x64x3
- **Loss:** Binary Crossentropy
- **Optimizer:** Adam
- **Output:** Sigmoid probability (binary classification)

---

## 🗂️ Project Structure

```
cnn/
├── app.py                   # Gradio app / inference script
├── pneumonia_model.keras    # Trained CNN model
├── requirements.txt         # Python dependencies
└── runtime.txt              # Python version pin for deployment
```

---

## ⚙️ Tech Stack

| Component        | Tool/Library         |
|------------------|----------------------|
| Model            | TensorFlow / Keras   |
| Web Interface    | Gradio               |
| Deployment       | Render (Free Tier)   |
| Language         | Python 3.11          |

---

## 🚀 Running Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/ArpanSingh1/pneumonia-classifier-app.git
   cd pneumonia-classifier-app/cnn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. Open the local URL shown in the terminal (usually `http://localhost:7860`).

---

## 🌐 Deployment (Render)

- **Root Directory:** `cnn`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python app.py`
- **Environment Variable:** `PYTHON_VERSION=3.11.9` (ensures TensorFlow compatibility)

---

## 📊 Model Performance

Trained on chest X-ray dataset with the following results during training:
- Validation Accuracy: ~97.5%
- Validation Loss: as low as ~0.05

---

## ⚠️ Disclaimer

This model is built for **educational/demonstration purposes only** and should **not** be used for real medical diagnosis. Always consult a certified medical professional for actual diagnosis.

---

## 👤 Author

**Arpan Singh**
- GitHub: [@ArpanSingh1](https://github.com/ArpanSingh1)
