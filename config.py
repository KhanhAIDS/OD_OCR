import streamlit as st
import cv2
import numpy as np

st.set_page_config(layout="wide", page_title="BOM OCR Preprocessing")
st.title("BOM OCR Preprocessing Test")

default_settings = {
    'use_clahe': True, 'contrast': 1.0, 'brightness': 0, 'sharpen': 0.0,
    'block_size': 15, 'c_val': 5,
    'morph_type': "None", 'morph_size': 2
}

for k, v in default_settings.items():
    if k not in st.session_state: st.session_state[k] = v

def reset_settings():
    for k, v in default_settings.items(): st.session_state[k] = v

uploaded_file = st.sidebar.file_uploader("1. Upload", type=["jpg", "jpeg", "png"])
st.sidebar.button("Reset", on_click=reset_settings)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)
    
    st.sidebar.header("2. Preprocessing configs")
    st.session_state.use_clahe = st.sidebar.checkbox("Use CLAHE", value=st.session_state.use_clahe)
    st.session_state.contrast = st.sidebar.slider("Contrast", 0.5, 3.0, st.session_state.contrast, 0.1)
    st.session_state.brightness = st.sidebar.slider("Brightness", -100, 100, st.session_state.brightness, 5)
    st.session_state.sharpen = st.sidebar.slider("Sharpen", 0.0, 3.0, st.session_state.sharpen, 0.2)
    
    st.sidebar.subheader("Sửa nét đứt/dính")
    st.session_state.morph_type = st.sidebar.selectbox("Morphology", ["None", "Dilate", "Erode"], index=["None", "Dilate", "Erode"].index(st.session_state.morph_type))
    st.session_state.morph_size = st.sidebar.slider("Kernel Size", 1, 7, st.session_state.morph_size, step=1)

    st.sidebar.subheader("Góc nhìn OCR (Adaptive Threshold)")
    st.session_state.block_size = st.sidebar.slider("Block Size", 3, 99, st.session_state.block_size, step=2)
    st.session_state.c_val = st.sidebar.slider("C", -20, 20, st.session_state.c_val)

    processed_img = img_rgb.copy()

    if st.session_state.use_clahe:
        lab = cv2.cvtColor(processed_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
        processed_img = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)

    if st.session_state.contrast != 1.0 or st.session_state.brightness != 0:
        processed_img = cv2.convertScaleAbs(processed_img, alpha=st.session_state.contrast, beta=st.session_state.brightness)

    if st.session_state.sharpen > 0:
        blurred = cv2.GaussianBlur(processed_img, (0, 0), 3.0)
        processed_img = cv2.addWeighted(processed_img, 1.0 + st.session_state.sharpen, blurred, -st.session_state.sharpen, 0)

    gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
    ocr_view = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, st.session_state.block_size, st.session_state.c_val)

    if st.session_state.morph_type != "None" and st.session_state.morph_size > 1:
        kernel = np.ones((st.session_state.morph_size, st.session_state.morph_size), np.uint8)
        inverted = cv2.bitwise_not(ocr_view)
        if "Dilate" in st.session_state.morph_type:
            inverted = cv2.dilate(inverted, kernel, iterations=1)
        elif "Erode" in st.session_state.morph_type:
            inverted = cv2.erode(inverted, kernel, iterations=1)
        ocr_view = cv2.bitwise_not(inverted)

    ocr_view_rgb = cv2.cvtColor(ocr_view, cv2.COLOR_GRAY2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Original")
        st.image(img_rgb, use_container_width=True)
    with col2:
        st.subheader("2. Góc nhìn OCR")
        st.image(ocr_view_rgb, use_container_width=True)

else:
    st.info("Upload diagrams")