import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
import random
import pandas as pd

# =========================================================
# Model
# =========================================================
MODEL_PATH = "digit_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model at {MODEL_PATH}: {e}")
    st.stop()

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# =========================================================
# Preprocessing utilities (robust for uploads)
# =========================================================
def _center_on_28x28(binary_img):
    # find bbox
    coords = np.where(binary_img > 0)
    if coords[0].size == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    # padding
    pad = max(5, min(20, min(x_max - x_min, y_max - y_min) // 4))
    y_min, y_max = max(0, y_min - pad), min(binary_img.shape[0] - 1, y_max + pad)
    x_min, x_max = max(0, x_min - pad), min(binary_img.shape[1] - 1, x_max + pad)
    crop = binary_img[y_min:y_max + 1, x_min:x_max + 1]

    # resize to 20x20 preserving aspect
    h, w = crop.shape
    if h == 0 or w == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # paste on 28x28
    final = np.zeros((28, 28), dtype=np.uint8)
    y_off, x_off = (28 - nh) // 2, (28 - nw) // 2
    final[y_off:y_off + nh, x_off:x_off + nw] = resized

    # center-of-mass shift (bounded)
    m = cv2.moments(final)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        dx, dy = int(14 - cx), int(14 - cy)
        dx, dy = np.clip([dx, dy], -5, 5)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        final = cv2.warpAffine(final, M, (28, 28), borderValue=0)
    return final

def preprocess_canvas(pil_img):
    arr = np.array(pil_img.convert("L"))
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    final = _center_on_28x28(thresh)
    final_norm = final.astype("float32") / 255.0
    return final_norm.reshape(1, 28, 28, 1), {
        "gray": cv2.resize(arr, (140, 140), interpolation=cv2.INTER_NEAREST),
        "threshold": cv2.resize(thresh, (140, 140), interpolation=cv2.INTER_NEAREST),
        "final28": cv2.resize(final, (140, 140), interpolation=cv2.INTER_NEAREST),
        "method": "Canvas-Otsu"
    }

def preprocess_mnist(img_array):
    final = img_array.astype("float32") / 255.0
    return final.reshape(1, 28, 28, 1), {
        "gray": cv2.resize(img_array, (140, 140), interpolation=cv2.INTER_NEAREST),
        "threshold": cv2.resize(img_array, (140, 140), interpolation=cv2.INTER_NEAREST),
        "final28": cv2.resize(img_array, (140, 140), interpolation=cv2.INTER_NEAREST),
        "method": "MNIST"
    }

def preprocess_upload(pil_img):
    """
    Ultra-robust pipeline for real-world uploaded paper photos:
    - Handles dark paper, shadows, and faint pen strokes
    - Removes background texture/noise
    - Keeps only the largest connected component (the digit)
    - Centers and normalizes to MNIST format
    """
    # 1) RGBA→RGB and grayscale
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    gray = np.array(pil_img.convert("L"))

    # 2) Robust contrast handling for dark pages
    mean_val = gray.mean()
    if mean_val < 120:
        # If very dark paper, invert so ink becomes bright
        if mean_val < 60:
            gray = 255 - gray
        gray = np.array(ImageOps.equalize(Image.fromarray(gray)))
        gray = np.clip((gray - gray.mean()) * 3.0 + gray.mean(), 0, 255).astype(np.uint8)
    else:
        gray = np.array(ImageEnhance.Contrast(Image.fromarray(gray)).enhance(2.0))

    # 3) Denoise background texture (paper grain) using bilateral + median
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=20, sigmaSpace=20)
    gray = cv2.medianBlur(gray, 3)

    # 4) Adaptive binarization tuned for paper photos (digit -> white)
    th_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,   # larger block-size better for uneven lighting
        -10   # negative C biases towards foreground (white)
    )

    # 5) Remove small speckle noise with morphology
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(th_adapt, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6) Keep only the largest connected component (digit), drop paper noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels <= 1:
        # fallback to threshold directly
        main = opened
        method = "Adaptive-Fallback"
    else:
        # skip label 0 (background); choose the biggest by area
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest_idx = 1 + np.argmax(areas)
        main = np.where(labels == biggest_idx, 255, 0).astype(np.uint8)
        method = "Adaptive+LargestCC"

    # 7) Final MNIST-style layout (digit should be white on black)
    final = _center_on_28x28(main)

    # 8) Normalize
    final_norm = final.astype("float32") / 255.0

    return final_norm.reshape(1, 28, 28, 1), {
        "gray": cv2.resize(gray, (140, 140), interpolation=cv2.INTER_NEAREST),
        "threshold": cv2.resize(main, (140, 140), interpolation=cv2.INTER_NEAREST),
        "final28": cv2.resize(final, (140, 140), interpolation=cv2.INTER_NEAREST),
        "method": method
    }

def preprocess(source_img, source="upload"):
    if source == "canvas":
        return preprocess_canvas(source_img)
    if source == "upload":
        return preprocess_upload(source_img)
    if source == "mnist":
        return preprocess_mnist(source_img)
    raise ValueError("Unknown source")

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✏️ Handwritten Digit Recognition")
st.write("Choose input method. Check the 'Final 28x28' preview to see exactly what the model sees.")

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

            st.info(f"Preprocessing method used: {preview.get('method', 'Standard')}")
            c1, c2, c3 = st.columns(3)
            c1.image(preview["gray"], caption="Enhanced Grayscale")
            c2.image(preview["threshold"], caption="Binarized (largest component)")
            c3.image(preview["final28"], caption="Final 28x28")

            pred = model.predict(final_input, verbose=0)[0]
            label = int(np.argmax(pred))
            confidence = float(np.max(pred))

            st.success(f"Prediction: {label} (Confidence: {confidence:.2%})")
            st.bar_chart(pd.Series(pred, index=range(10)))

            with st.expander("Preprocessing Details"):
                final_28 = (final_input[0, :, :, 0] * 255).astype(np.uint8)
                fg = int((final_28 > 0).sum())
                st.write(f"Foreground pixels: {fg}/784 ({fg/784*100:.1f}%)")
                st.write(f"Intensity range: [{final_input.min():.3f}, {final_input.max():.3f}]")

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
