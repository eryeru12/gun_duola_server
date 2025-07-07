import cv2
import numpy as np
import os
import torch
import tensorflow as tf

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from keras.models import load_model
import keras 

class SRCNN_ESRGAN:


    @staticmethod
    def load_esrgan_model(model_path):
        """
        Load the ESRGAN model from the specified path.
        """
        if not model_path or not os.path.exists(model_path):
            print("ESRGAN model file not found.")
            return None
        try:
            # Load PyTorch model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.load(model_path, map_location=device)
            model.eval()
            print(f"Successfully loaded ESRGAN model from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load ESRGAN model: {str(e)}")
            return None
        
    @staticmethod
    def preprocess_image(image, scale=3):
        """
        Preprocess the image for SRCNN model.
        Resize and normalize the image.
        """
        height, width, _ = image.shape

        image = cv2.resize(image, (width//scale, height//scale), interpolation=cv2.INTER_CUBIC) 
        
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)  # Resize back to original size   

        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        return image

    @staticmethod
    def postprocess_image_esrgan(image):
        """
        Postprocess the image after ESRGAN prediction.
        Convert back to uint8 and scale to [0, 255].
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(image, axis=0)  # Add batch dimension

    @staticmethod
    def adjust_contrast_brightness(image, factor=1.0, brightness=0):
        """
        Adjust the contrast of the image.
        """
        
        return cv2.convertScaleAbs(image, alpha=factor, beta=brightness)

    @staticmethod
    def sharpen_image(image, sigma=1.0):
        """
        Sharpen the image using a Gaussian kernel.
        """
        kernel = np.array([[0, -1, 0],
                        [-1, 5 , -1],
                        [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def super_resolve_image_srcnn(model, image, scale=3):
        """
        Apply super-resolution using the ESRGAN model.
        """
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        output = model.predict(image)

        output = output[0]*255.0    
        
        output = np.clip(output, 0, 255)  # Ensure pixel values are in [0, 255]

        output = output.astype(np.uint8)  # Convert to uint8

        return output

    @staticmethod
    def super_resolve_image_esrgan(model, image, scale=4):
        """
        Apply super-resolution using the ESRGAN model.
        """
        # Convert image to tensor
        img_tensor = ToTensor()(image).unsqueeze(0)
        
        # Run prediction
        with torch.no_grad():
            output = model(img_tensor)
        
        # Convert back to numpy array
        output = output.squeeze().permute(1, 2, 0).numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output
