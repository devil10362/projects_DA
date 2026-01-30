import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained emotion recognition model
model = load_model("emotion.h5")   # <-- export from recognition.ipynb

# Define class labels (adjust these based on your notebook training)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

IMG_SIZE = (48, 48)  # typical size for FER models (change if your notebook used different)

def predict_emotion(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # many FER models use grayscale
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Prediction
    preds = model.predict(img, verbose=0)[0]
    emotion_idx = np.argmax(preds)
    return emotion_labels[emotion_idx]

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        emotion = predict_emotion(file_path)
        messagebox.showinfo("Emotion Recognition", f"Predicted Emotion: {emotion}")

# ---------------- Tkinter UI ----------------
root = tk.Tk()
root.title("Emotion Recognition")
root.geometry("400x250")

title = tk.Label(root, text="Face Emotion Recognition", font=("Arial", 16, "bold"))
title.pack(pady=20)

upload_btn = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_btn.pack(pady=20)

exit_btn = tk.Button(root, text="Exit", command=root.destroy, font=("Arial", 12))
exit_btn.pack(pady=10)

root.mainloop()
