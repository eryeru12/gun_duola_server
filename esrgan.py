import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import torch

class ESRGAN:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                #self.model = tf.keras.models.load_model(model_path)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model = torch.load(model_path, map_location=device)
                #self.model = torch.jit.load(model_path)  # Load as Torch model
                #self.model.eval()  # Set to evaluation mode
                print(f"Successfully loaded ESRGAN model from {model_path}")
            except Exception as e:
                print(f"Failed to load ESRGAN model: {str(e)}")
                self.model = None

    def preprocess_image(self, img):
        # Convert to RGB and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def postprocess_image(self, img):
        # Scale back to 0-255 range
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def predict(self, img, scale=4):
        if not self.model:
            return img
            
        # Preprocess
        img = self.preprocess_image(img)
        
        # Pad image to be divisible by scale factor
        h, w = img.shape[:2]
        pad_h = (scale - h % scale) % scale
        pad_w = (scale - w % scale) % scale
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Predict
        sr_img = self.model.predict(img[np.newaxis, ...], verbose=0)[0]
        
        # Remove padding
        sr_img = sr_img[:h*scale, :w*scale]
        
        # Postprocess
        return self.postprocess_image(sr_img)

# Pre-trained model URL (RRDB_PSNR_x4.pth from ESRGAN)
PRETRAINED_MODEL_URL = "https://github.com/xinntao/ESRGAN/releases/download/v0.1.1/RRDB_PSNR_x4.pth"
