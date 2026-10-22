import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

# Configuração do visual do site
st.set_page_config(page_title="AI Vision Analytics", layout="wide")
st.title("👁️ AI Vision Analytics - SENAI")
st.markdown("Visual processing platform")

# Endereços da nossa API
URL_API_IMAGE = "http://127.0.0.1:8000/detect_objects"


# ==========================================
# ABA 1: FRONT-END DA IMAGE
# ==========================================

st.sidebar.header("Inference Configuration")

model_choice = st.sidebar.selectbox(
    "Select Model", 
    options=["yolov8n", "yolo26n", "yolo26s-1280"],
    index=0,
    help="Choose the model for object detection.")


st.header("Image Analysis")

arquivo_img = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"], key="img")

if arquivo_img is not None:
    image = Image.open(arquivo_img)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
        
    if st.button("🚀 Start Image Analysis", use_container_width=True):
        with st.spinner('Acessing AI Server...'):
            
            # CORRECTION: If the image has a transparency channel (RGBA/PNG), convert to pure RGB
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            # Prepare and send to the API
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            response = requests.post(
                URL_API_IMAGE, 
                files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
                data={"model_name": model_choice}
                )
            
            if response.status_code == 200:
                dados = response.json()
                
                # Draw bounding boxes on the image
                annotated_img = image.copy()
                draw = ImageDraw.Draw(annotated_img)
                
                for obj in dados.get('detected_objects', []):
                    c = obj['coordinates']
                    label = obj['label']
                    conf = int(obj['confidence'] * 100)
                    
                    # Coordinates x1, y1, x2, y2
                    draw.rectangle([c['x1'], c['y1'], c['x2'], c['y2']], outline="red", width=4)
                    draw.text((c['x1'], max(0, c['y1'] - 12)), f"{label} {conf}%", fill="red")

                with col2:
                    st.image(annotated_img, caption=f"Found {dados['total_objects']} objects!", use_container_width=True)
                
                st.success("JSON returned by the API:")
                st.json(dados)
            else:
                st.error("Error on API server.")
