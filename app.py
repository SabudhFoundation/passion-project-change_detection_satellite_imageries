from fastapi import FastAPI
import tensorflow as tf
import keras
import rasterio
import geopandas
import albumentations
import cv2
import sklearn

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Docker Container Running Successfully",
        "TensorFlow": tf.__version__,
        "Keras": keras.__version__,
        "OpenCV": cv2.__version__,
        "GPU Available": str(tf.config.list_physical_devices('GPU')),
        "Rasterio": "Installed",
        "GeoPandas": "Installed",
        "Albumentations": "Installed",
        "Scikit-learn": "Installed"
    }

@app.get("/health")
def health():
    return {
        "status": "OK"
    }