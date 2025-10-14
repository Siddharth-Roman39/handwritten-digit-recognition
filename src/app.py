# app.py
"""
Unified Streamlit app for MNIST (English) + Devanagari digit recognition.
Features:
 - Language selector (MNIST or Devanagari)
 - Canvas draw + Upload + Random MNIST sample
 - Interactive preprocessing controls (thresholding, blur, invert, morphology)
 - Debug previews: grayscale, binarized, final resized input
 - Top-3 predictions shown with confidences
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance
import random
import pandas as pd
import os
import math

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Digit Recognizer — MNIST + Devanagari", layout="centered")
st.title("🧠 Digit Recognizer — MNIST & Devanagari")
st.markdown(
    "Choose language, then draw/upload a digit. Use the preprocessing controls if predictions look wrong."
)

# Default model paths (edit if yours are elsewhere)
MNIST_MODEL_PATH = "digit_model.keras"
DEVANAGARI_MODEL_PATH = "devanagari_digit_model.keras"

# ---------------------------
# UI: Model selection & load
# ---------------------------
lang = st.selectbox("Select model / digit type:", ("English (MNIST)", "Devanagari"))

if lang == "English (MNIST)":
    MODEL_PATH = MNIST_MODEL_PATH
    IMG_SIZE = 28
    LABELS = [str(i) for i in range(10)]
    has_random = True
else:
    MODEL_PATH = DEVANAGARI_MODEL_PATH
    IMG_SIZE = 32
    # Devanagari digits U+0966..U+096F (० १ २ ३ ४ ५ ६ ७ ८ ९)
    LABELS = [chr(0x0966 + i) for i in range(10)]
    has_random = False

# Attempt to load model
model_load_error = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success(f"Loaded model: `{MODEL_PATH}`")
except Exception as e:
    model = None
    model_load_error = e
    st.error(f"Could not load model at `{MODEL_PATH}`: {e}")
    st.stop()

# If MNIST, load test dataset for random sample
if lang == "English (MNIST)":
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    x_test = y_test = None

# ---------------------------
# UI: Input mode
# ---------------------------
modes = ["Draw on Canvas", "Upload Image"]
if has_random:
    modes.append("Random MNIST Sample")
mode = st.radio("Choose input method:", modes)

# ---------------------------
# UI: Preprocessing controls (exposed to user for tuning)
# ---------------------------
st.sidebar.header("Preprocessing controls (tweak to improve predictions)")

# General
invert_guess = st.sidebar.checkbox("Auto-detect & invert (white bg -> black bg)", value=True)
normalize_to = st.sidebar.selectbox("Normalization type", ("0-1 (float)", "0-255 (uint8)"), index=0)

# Denoise
use_bilateral = st.sidebar.checkbox("Use bilateral filter (denoise)", value=True)
bilateral_d = st.sidebar.slider("bilateral d (0=auto)", 0, 15, 7)
use_median = st.sidebar.checkbox("Use median blur", value=True)
median_k = st.sidebar.slider("median kernel size (odd)", 1, 7, 3)
if median_k % 2 == 0:
    median_k += 1

# Thresholding
th_method = st.sidebar.selectbox("Thresholding method:", ("Otsu (auto)", "Adaptive (Gaussian)", "Fixed"))
fixed_thresh = st.sidebar.slider("Fixed threshold value", 50, 200, 128)
adaptive_block = st.sidebar.slider("Adaptive block size (odd)", 11, 51, 31)
if adaptive_block % 2 == 0:
    adaptive_block += 1
adaptive_C = st.sidebar.slider("Adaptive C (bias)", -20, 20, -10)

# Morphology & cleanup
use_morph = st.sidebar.checkbox("Apply morphological opening (remove speckles)", value=True)
morph_iter = st.sidebar.slider("Morph iterations", 0, 3, 1)

# Final tuning
center_shift_clip = st.sidebar.slider("Center shift clip (px)", 0, 6, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("If predictions are unreliable, try toggling invert / threshold or increase denoising.")

# ---------------------------
# Helper functions
# ---------------------------
def pil_to_gray_numpy(pil: Image.Image) -> np.ndarray:
    """Return uint8 grayscale numpy array from PIL (handles RGBA)"""
    if pil.mode == "RGBA":
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        bg.paste(pil, mask=pil.split()[3])  # 3 is alpha
        pil = bg
    if pil.mode != "L":
        pil = pil.convert("L")
    arr = np.array(pil).astype(np.uint8)
    return arr

def auto_invert_if_needed(gray: np.ndarray) -> (np.ndarray, bool):
    """Flip colors if background is white (heuristic using mean)"""
    mean = gray.mean()
    if invert_guess and mean > 127:  # bright background -> likely black strokes on white bg -> invert
        return 255 - gray, True
    return gray, False

def denoise(gray: np.ndarray) -> np.ndarray:
    g = gray.copy()
    if use_bilateral:
        d = bilateral_d if bilateral_d > 0 else 7
        g = cv2.bilateralFilter(g, d, sigmaColor=75, sigmaSpace=75)
    if use_median and median_k > 1:
        g = cv2.medianBlur(g, median_k)
    return g

def threshold_image(gray: np.ndarray) -> (np.ndarray, str):
    """Apply thresholding based on chosen method, return binary (0/255)"""
    if th_method == "Otsu (auto)":
        # Otsu with inversion such that digit is white (255)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # We want digit=white. If the binary has background white and digit black, invert.
        # Determine which is foreground by comparing counts near center
        if np.sum(th == 255) > np.sum(th == 0):
            # if majority is white, invert so digit becomes white on black
            th = 255 - th
        return th, "Otsu"
    elif th_method == "Adaptive (Gaussian)":
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_block, adaptive_C
        )
        # ensure digit is white
        if np.sum(th == 255) > np.sum(th == 0):
            th = 255 - th
        return th, "Adaptive"
    else:  # Fixed
        _, th = cv2.threshold(gray, fixed_thresh, 255, cv2.THRESH_BINARY)
        # ensure digit is white
        if np.sum(th == 255) > np.sum(th == 0):
            th = 255 - th
        return th, "Fixed"

def keep_largest_component(binary: np.ndarray) -> np.ndarray:
    """Return binary image keeping only the largest connected component (helps remove paper noise)"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + np.argmax(areas)
    mask = (labels == biggest).astype(np.uint8) * 255
    return mask

def center_and_resize(binary: np.ndarray, size: int) -> np.ndarray:
    """
    Place the binary (0/255) digit into a size x size box, preserving aspect,
    leaving ~4 px margin (like MNIST).
    """
    # find bbox of white pixels
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    crop = binary[y_min:y_max+1, x_min:x_max+1]

    h, w = crop.shape
    # target box for digit is size - 8 (4 px margin around) — similar to MNIST style
    box = max(8, size - 8)
    scale = box / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    final = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - nh) // 2
    x_off = (size - nw) // 2
    final[y_off:y_off+nh, x_off:x_off+nw] = resized

    # Center of mass shift (bounded by clip)
    M = cv2.moments(final)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dx = int(size//2 - cx)
        dy = int(size//2 - cy)
        dx, dy = np.clip([dx, dy], -center_shift_clip, center_shift_clip)
        T = np.float32([[1, 0, dx], [0, 1, dy]])
        final = cv2.warpAffine(final, T, (size, size), borderValue=0)
    return final

def prepare_for_model(final_img: np.ndarray) -> np.ndarray:
    """Return model-ready array shape (1, size, size, 1) with normalization matching training."""
    arr = final_img.astype(np.float32)
    if normalize_to == "0-1 (float)":
        arr = arr / 255.0
    else:
        # keep 0-255 but cast to float32 (some models trained with rescale=1./255; still OK)
        arr = arr  # could leave as 0-255
    arr = arr.reshape(1, final_img.shape[0], final_img.shape[1], 1)
    return arr

def predict_topk(inp_arr: np.ndarray, k: int = 3):
    pred = model.predict(inp_arr, verbose=0)[0]
    top_idx = np.argsort(pred)[::-1][:k]
    return [(LABELS[i], float(pred[i])) for i in top_idx], pred

# ---------------------------
# Processing pipeline wrapper
# ---------------------------
def preprocess_pipeline(pil: Image.Image, size: int):
    """Takes a PIL image and returns:
       - preview dict with 'gray', 'binarized', 'final_resized'
       - model_input array (1,size,size,1)
       - debug info dict
    """
    gray = pil_to_gray_numpy(pil)
    gray0 = gray.copy()
    gray, inverted = auto_invert_if_needed(gray)
    den = denoise(gray)
    bin_img, method_name = threshold_image(den)
    if use_morph and morph_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
    # Keep largest component to drop paper texture
    cleaned = keep_largest_component(bin_img)
    final = center_and_resize(cleaned, size)
    model_in = prepare_for_model(final)
    preview = {
        "gray": cv2.resize(gray0, (140, 140), interpolation=cv2.INTER_NEAREST),
        "denoised": cv2.resize(den, (140, 140), interpolation=cv2.INTER_NEAREST),
        "binarized": cv2.resize(bin_img, (140, 140), interpolation=cv2.INTER_NEAREST),
        "cleaned": cv2.resize(cleaned, (140, 140), interpolation=cv2.INTER_NEAREST),
        "final": cv2.resize(final, (140, 140), interpolation=cv2.INTER_NEAREST),
        "method": method_name,
        "inverted": inverted
    }
    debug = {
        "orig_mean": float(gray0.mean()),
        "inverted": inverted,
        "method": method_name
    }
    return preview, model_in, debug

# ---------------------------
# UI: Input handling
# ---------------------------
def show_prediction_blocks(pred_list, prob_array):
    df = pd.DataFrame({
        "label": [p[0] for p in pred_list],
        "prob": [p[1] for p in pred_list]
    })
    st.table(df.assign(confidence=lambda d: (d.prob * 100).round(2).astype(str) + "%"))
    # full probability bar chart for all labels
    st.bar_chart(pd.Series(prob_array, index=LABELS))

if mode == "Draw on Canvas":
    st.subheader("Draw a digit (use thick stroke for better results)")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=18,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{lang}"
    )
    if st.button("Predict from Drawing"):
        if canvas_result.image_data is None:
            st.warning("Please draw something on the canvas.")
        else:
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
            # convert to RGB to remove alpha, pipeline will convert to grayscale
            img = img.convert("RGB")
            preview, model_in, debug = preprocess_pipeline(img, IMG_SIZE)
            st.markdown("**Preview (what the model sees):**")
            cols = st.columns(5)
            cols[0].image(preview["gray"], caption="Original Gray")
            cols[1].image(preview["denoised"], caption="Denoised")
            cols[2].image(preview["binarized"], caption=f"Binarized ({preview['method']})")
            cols[3].image(preview["cleaned"], caption="Largest CC kept")
            cols[4].image(preview["final"], caption=f"Final {IMG_SIZE}×{IMG_SIZE}")
            preds, probs = predict_topk(model_in, k=3)
            st.success(f"Top prediction: {preds[0][0]} ({preds[0][1]*100:.2f}%)")
            show_prediction_blocks(preds, probs)

elif mode == "Upload Image":
    st.subheader("Upload an image (photo or scan). Good lighting helps.")
    uploaded = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", width=220)
        if st.button("Predict from Upload"):
            preview, model_in, debug = preprocess_pipeline(img, IMG_SIZE)
            st.markdown("**Preview (what the model sees):**")
            cols = st.columns(5)
            cols[0].image(preview["gray"], caption="Original Gray")
            cols[1].image(preview["denoised"], caption="Denoised")
            cols[2].image(preview["binarized"], caption=f"Binarized ({preview['method']})")
            cols[3].image(preview["cleaned"], caption="Largest CC kept")
            cols[4].image(preview["final"], caption=f"Final {IMG_SIZE}×{IMG_SIZE}")
            preds, probs = predict_topk(model_in, k=3)
            st.success(f"Top prediction: {preds[0][0]} ({preds[0][1]*100:.2f}%)")
            show_prediction_blocks(preds, probs)
            with st.expander("Debug info"):
                st.write(debug)

elif mode == "Random MNIST Sample":
    st.subheader("Random MNIST Test Sample (no preprocessing)")
    if st.button("Generate Random MNIST Sample"):
        idx = random.randint(0, len(x_test) - 1)
        img = x_test[idx]
        st.image(img, caption=f"Ground Truth: {y_test[idx]}", width=200)
        arr = img.reshape(1, 28, 28, 1).astype(np.float32) / 255.0
        preds, probs = predict_topk(arr, 3)
        st.info(f"Prediction: {preds[0][0]} ({preds[0][1]*100:.2f}%)")
        show_prediction_blocks(preds, probs)

# ---------------------------
# Helpful tips & quick actions
# ---------------------------
st.markdown("---")
st.markdown("**Troubleshooting tips:**")
st.markdown(
    "- If predictions are wrong: try toggling **Auto invert**, or switch threshold method (Otsu/Adaptive/Fixed).  \n"
    "- Increase denoising (bilateral / median) for photographed images with texture.  \n"
    "- If strokes are very thin, increase stroke width in the canvas or use a thicker pen.  \n"
    "- For Devanagari digits, ensure your model path is correct and that labels align with your training labels."
)

st.markdown("**Developer:** If you want, I can produce a `requirements.txt` for this environment (recommended: `opencv-python-headless`, `tensorflow`, `streamlit`, `streamlit-drawable-canvas`).")
