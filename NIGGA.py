import streamlit as st
from roboflow import Roboflow
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# ====================
# 🔧 Configuración Roboflow
# ====================
API_KEY = "uSCqi2uF8qf6Udwu1sm0"
WORKSPACE = "hx-hezqh"
PROJECT = "ppe-detection-yfmym"
VERSION = 1

# ====================
# 🎨 Interfaz Streamlit
# ====================
st.set_page_config(page_title="🚨 Sistema de Detección de EPP", layout="centered")
st.title("🚨 Sistema de Detección de Equipo de Protección Personal")
st.markdown("---")

st.sidebar.header("⚙️ Configuración")
fuente = st.sidebar.radio("📷 Seleccione fuente de imagen:", ["Subir imagen", "Usar cámara"])

st.markdown("## 🛡️ Elementos Requeridos")
st.markdown("Este sistema detecta si una persona está usando casco, chaleco, gafas, etc.")

# ====================
# 📷 Captura de imagen
# ====================
img = None

if fuente == "Subir imagen":
    st.markdown("### 🖼️ Seleccione imagen:")
    file = st.file_uploader("Cargue una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
        st.image(img, caption="Imagen cargada", use_column_width=True)

elif fuente == "Usar cámara":
    st.markdown("### 📸 Capture una imagen:")
    img_bytes = st.camera_input("Tomar foto")
    if img_bytes:
        img = Image.open(img_bytes)
        st.image(img, caption="Imagen capturada", use_column_width=True)

# ====================
# 🔍 Procesamiento con Roboflow
# ====================
if img:
    st.markdown("## 🔍 Resultados del Análisis")

    # Guardar imagen temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        img.save(temp_file.name)
        image_path = temp_file.name

    # Inicializar Roboflow
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    model = project.version(VERSION).model

    # Realizar predicción
    with st.spinner("Analizando imagen..."):
        result = model.predict(image_path, confidence=40, overlap=30).json()

    # Mostrar resultados
    detections = result.get("predictions", [])
    if detections:
        st.success(f"Se detectaron {len(detections)} objetos:")
        for det in detections:
            st.markdown(f"- **{det['class']}** con {round(det['confidence']*100, 2)}% de confianza")
        
        # Mostrar imagen con cajas
        output_path = model.predict(image_path, confidence=40, overlap=30).save(output_dir=".")
        st.image(output_path, caption="Detecciones", use_column_width=True)

        # Borrar imagen temporal
        os.remove(image_path)
        if os.path.exists("prediction.jpg"):
            os.remove("prediction.jpg")
    else:
        st.warning("No se detectaron elementos de protección en la imagen.")

else:
    st.info("ℹ️ Configure los parámetros y cargue una imagen para realizar el análisis.")
