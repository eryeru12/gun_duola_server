import tensorflow as tf
import numpy as np
import cv2

class SRCNN:
    def __init__(self, weights_path=None):
        self.model = self.build_model()
        if weights_path:
            self.model.load_weights(weights_path)

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 3)),
            tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(3, (5, 5), padding='same')
        ])
        return model

    def preprocess_image(self, img):
        # Convert BGR to YCrCb and extract Y channel
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = img_ycrcb[:,:,0].astype(np.float32) / 255.0
        return y, img_ycrcb

    def postprocess_image(self, y_pred, img_ycrcb):
        # Scale back to 0-255 range
        y_pred = np.clip(y_pred * 255.0, 0, 255).astype(np.uint8)
        # Merge with original CrCb channels
        img_ycrcb[:,:,0] = y_pred
        # Convert back to BGR
        return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

    def predict(self, img):
        # Preprocess
        y, img_ycrcb = self.preprocess_image(img)
        
        # Predict
        y_pred = self.model.predict(y[np.newaxis,...,np.newaxis], verbose=0)[0,...,0]
        
        # Postprocess
        return self.postprocess_image(y_pred, img_ycrcb)

# Pre-trained weights URL (to be downloaded)
PRETRAINED_WEIGHTS_URL = "https://github.com/MarkPrecursor/SRCNN-keras/releases/download/v0.1/srcnn_weights.h5"
