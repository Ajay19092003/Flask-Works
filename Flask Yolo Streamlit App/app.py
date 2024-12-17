import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5x")  # replace 'yolov5s' with 'yolov5m', 'yolov5l', or 'yolov5x' for larger models
    return model


def detect_objects(model, uploaded_image):

    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    results = model(image)
    results.render() 

    st.image(results.ims[0], caption="Processed Image", use_column_width=True)
    detected_classes = results.names
    st.write("Detected classes:", detected_classes)

def main():
    
    st.title("YOLOv5 Object Detection")

    st.write(
        "Upload an image to perform object detection using the YOLOv5 model."
    )

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_image is not None:
        if uploaded_image.type.startswith("image"):
        
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            model = load_model() 
            detect_objects(model, uploaded_image)  
        elif uploaded_image.type.startswith("video"):
            
            st.video(uploaded_image)
            st.write("Video object detection will be processed in future versions.")
        else:
            st.error("Unsupported file type. Please upload an image or video.")

if __name__ == "__main__":
    main()
