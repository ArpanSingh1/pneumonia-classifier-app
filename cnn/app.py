import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

model = tf.keras.models.load_model("pneumonia_model.keras")

IMG_SIZE = (64, 64)  # adjust to whatever size you trained on
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # adjust order to match your training

def predict(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    if len(preds) == 1:  # sigmoid binary output
        prob = float(preds[0])
        return {CLASS_NAMES[1]: prob, CLASS_NAMES[0]: 1 - prob}
    else:  # softmax
        return {CLASS_NAMES[i]: float(preds[i]) for i in range(len(preds))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Pneumonia Classifier"
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
