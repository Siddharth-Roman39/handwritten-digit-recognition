import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random
import pandas as pd

# ---------------------------
# Load model
# ---------------------------
MODEL_PATH = "src/digit_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}: {e}")
    st.stop()

# Load MNIST test set
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# ---------------------------
# Preprocessing functions
# ---------------------------
def preprocess_canvas(pil_img):
    """Preprocessing for canvas drawings (clean black strokes on white)."""
    arr = np.array(pil_img.convert("L"))  # grayscale
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Crop
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]
    else:
        digit = thresh

    # Resize to 20x20
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Place on 28x28
    final = np.zeros((28, 28), dtype=np.uint8)
    x_off, y_off = (28 - new_w) // 2, (28 - new_h) // 2
    final[y_off:y_off+new_h, x_off:x_off+new_w] = digit_resized

    # Center shift
    m = cv2.moments(final)
    if m["m00"] != 0:
        cx, cy = m["m10"]/m["m00"], m["m01"]/m["m00"]
        dx, dy = int(14 - cx), int(14 - cy)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        final = cv2.warpAffine(final, M, (28, 28), borderValue=0)

    final_norm = final.astype("float32") / 255.0
    return final_norm.reshape(1, 28, 28, 1), {
        "gray": cv2.resize(arr, (140, 140), interpolation=cv2.INTER_NEAREST),
        "threshold": cv2.resize(thresh, (140, 140), interpolation=cv2.INTER_NEAREST),
        "final28": cv2.resize(final, (140, 140), interpolation=cv2.INTER_NEAREST),
    }


def preprocess_upload(pil_img):
    """Preprocessing for uploaded images (noisy, shadows, faint strokes)."""
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg

    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
    arr_rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)

    # Threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(otsu) < 10:
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15, 5
        )
    else:
        thresh = otsu

    # Morphology
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Crop
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = thresh[y:y+h, x:x+w]
    else:
        digit = thresh

    # Resize to 20x20
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Place on 28x28
    final = np.zeros((28, 28), dtype=np.uint8)
    x_off, y_off = (28 - new_w) // 2, (28 - new_h) // 2
    final[y_off:y_off+new_h, x_off:x_off+new_w] = digit_resized

    # Center shift
    m = cv2.moments(final)
    if m["m00"] != 0:
        cx, cy = m["m10"]/m["m00"], m["m01"]/m["m00"]
        dx, dy = int(14 - cx), int(14 - cy)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        final = cv2.warpAffine(final, M, (28, 28), borderValue=0)

    final_norm = final.astype("float32") / 255.0
    return final_norm.reshape(1, 28, 28, 1), {
        "gray": cv2.resize(gray, (140, 140), interpolation=cv2.INTER_NEAREST),
        "threshold": cv2.resize(thresh, (140, 140), interpolation=cv2.INTER_NEAREST),
        "final28": cv2.resize(final, (140, 140), interpolation=cv2.INTER_NEAREST),
    }


def preprocess_mnist(img_array):
    """Preprocessing for MNIST test images (already clean)."""
    final = img_array.astype("float32") / 255.0
    final_input = final.reshape(1, 28, 28, 1)
    return final_input, {
        "gray": cv2.resize(img_array, (140, 140), interpolation=cv2.INTER_NEAREST),
        "threshold": cv2.resize(img_array, (140, 140), interpolation=cv2.INTER_NEAREST),
        "final28": cv2.resize(img_array, (140, 140), interpolation=cv2.INTER_NEAREST),
    }


# Unified wrapper
def preprocess(pil_img_or_array, source="upload"):
    if source == "canvas":
        return preprocess_canvas(pil_img_or_array)
    elif source == "upload":
        return preprocess_upload(pil_img_or_array)
    elif source == "mnist":
        return preprocess_mnist(pil_img_or_array)
    else:
        raise ValueError("Unknown source type.")


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("✏️ Handwritten Digit Recognition")
st.write("Choose input method. Check the 'Final 28x28' preview to see what the model sees.")

choice = st.radio("Choose input method:", ("Draw on Canvas", "Upload Image", "Random MNIST Test Image"))

# ---------- Canvas ----------
if choice == "Draw on Canvas":
    st.subheader("Draw a digit (0–9)")

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=18,
        stroke_color="black",
        background_color="white",
        height=240,
        width=240,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict from Drawing"):
        if canvas_result.image_data is None:
            st.warning("Draw something first.")
        else:
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
            final_input, preview = preprocess(img, source="canvas")

            c1, c2, c3 = st.columns(3)
            c1.image(preview["gray"], caption="Grayscale")
            c2.image(preview["threshold"], caption="Binarized")
            c3.image(preview["final28"], caption="Final 28x28")

            pred = model.predict(final_input, verbose=0)[0]
            label = int(np.argmax(pred))
            st.success(f"Prediction: {label}")
            st.bar_chart(pd.Series(pred, index=range(10)))

# ---------- Upload ----------
elif choice == "Upload Image":
    st.subheader("Upload a digit image")

    uploaded = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded", width=200)

        if st.button("Predict from Upload"):
            final_input, preview = preprocess(img, source="upload")

            c1, c2, c3 = st.columns(3)
            c1.image(preview["gray"], caption="Grayscale")
            c2.image(preview["threshold"], caption="Binarized")
            c3.image(preview["final28"], caption="Final 28x28")

            pred = model.predict(final_input, verbose=0)[0]
            label = int(np.argmax(pred))
            st.success(f"Prediction: {label}")
            st.bar_chart(pd.Series(pred, index=range(10)))

# ---------- Random MNIST ----------
else:
    st.subheader("Random MNIST test sample")

    if st.button("Generate Random Test Image"):
        idx = random.randint(0, len(x_test) - 1)
        img = x_test[idx]
        st.image(img, caption=f"Actual: {y_test[idx]}", width=200)

        final_input, preview = preprocess(img, source="mnist")

        c1, c2, c3 = st.columns(3)
        c1.image(preview["gray"], caption="Grayscale")
        c2.image(preview["threshold"], caption="Binarized")
        c3.image(preview["final28"], caption="Final 28x28")

        pred = model.predict(final_input, verbose=0)[0]
        label = int(np.argmax(pred))
        st.info(f"Prediction: {label}")
        st.bar_chart(pd.Series(pred, index=range(10)))
