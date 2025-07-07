import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from collections import OrderedDict

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = lambda: RRDB(nf, gc=gc)
        
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB_block_f() for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

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
            # Create model instance
            model = RRDBNet()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'params' in checkpoint:
                model.load_state_dict(checkpoint['params'], strict=True)
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=True)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                model.load_state_dict(checkpoint, strict=True)
                
            model.eval()
            model.to(device)
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
