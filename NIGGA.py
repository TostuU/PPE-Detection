import streamlit as st
from PIL import Image
import cv2
import numpy as np
from roboflow import Roboflow

# Inicializar Roboflow
rf = Roboflow(api_key="uSCqi2uF8qf6Udwu1sm0")
project = rf.workspace().project("ppe-detection-yfmym")
model = project.version(1).model

# Imagen decorativa superior
st.image("https://i.pinimg.com/1200x/08/e0/c1/08e0c18e38e81d330ee1ea03bb795f32.jpg", use_column_width=True)

# Título principal
st.markdown("## 🦺 Sistema de Detección de EPP - Versión Mejorada 🔍")
st.markdown("---")

# Configuración
st.sidebar.markdown("## ⚙️ Configuración")
source = st.sidebar.radio("📷 Selecciona fuente de imagen:", ["Subir imagen", "Usar cámara"])

# Elementos requeridos (fijo o personalizable)
st.sidebar.markdown("## 🛡️ Elementos Requeridos")
st.sidebar.text("Casco, chaleco, guantes, gafas...")

# Cargar imagen
image = None
if source == "Subir imagen":
    uploaded_file = st.file_uploader("🖼️ Sube una imagen para analizar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
elif source == "Usar cámara":
    image = st.camera_input("📸 Captura una imagen")

# Mostrar resultados
st.markdown("## 🔍 Resultados del Análisis")
if image is not None:
    if source == "Usar cámara":
        image = Image.open(image)
        image.save("captured_image.jpg")
        prediction = model.predict("captured_image.jpg", confidence=40, overlap=30).json()
        result_image = model.predict("captured_image.jpg", confidence=40, overlap=30).save("result.jpg")
    else:
        cv2.imwrite("uploaded_image.jpg", image)
        prediction = model.predict("uploaded_image.jpg", confidence=40, overlap=30).json()
        result_image = model.predict("uploaded_image.jpg", confidence=40, overlap=30).save("result.jpg")

    st.image("result.jpg", caption="🔍 Resultado del modelo", use_column_width=True)

    st.success("✅ Análisis completo. Revisa la imagen con los elementos detectados.")

else:
    st.info("👈 Configura los parámetros y carga una imagen para analizarla.")

# Footer cool
st.markdown("---")
st.markdown("Hecho con 💻 por Cristian – Potenciado con [Roboflow](https://roboflow.com) 🚀")
