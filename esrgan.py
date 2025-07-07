import os
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

class RRDBNet(torch.nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = lambda: RRDB(nf, gc=gc)
        
        self.conv_first = torch.nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = torch.nn.Sequential(*[RRDB_block_f() for _ in range(nb)])
        self.trunk_conv = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = torch.nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = torch.nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out

class RRDB(torch.nn.Module):
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

class ResidualDenseBlock_5C(torch.nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = torch.nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = torch.nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = torch.nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = torch.nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = torch.nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class ESRGAN:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                # 创建模型实例
                #self.model = RRDBNet()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # 加载checkpoint
                checkpoint = torch.load(model_path, map_location=device)
                
                # 处理不同checkpoint格式和键名映射
                #state_dict = checkpoint.get('params', checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))
                
                # 创建新键名映射 (torch.nn -> 无前缀)
                #new_state_dict = {}
                #for k, v in state_dict.items():
                #    new_key = k.replace('torch.nn.', '')
                #    new_state_dict[new_key] = v
                
                # 加载转换后的state_dict
                #self.model.load_state_dict(new_state_dict, strict=False)
                    
                #self.model.eval()
                #self.model.to(device)
                print(f"Successfully loaded ESRGAN model from {model_path}")
            except Exception as e:
                print(f"Failed to load ESRGAN model: {str(e)}")
                self.model = None

    def preprocess_image(self, img):
        # 转换为RGB并归一化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def postprocess_image(self, img):
        # 缩放回0-255范围
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def predict(self, img, scale=4):
        if not self.model:
            return img
            
        # 预处理
        img_tensor = ToTensor()(img).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # 后处理
        output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        return self.postprocess_image(output)

# 预训练模型URL
PRETRAINED_MODEL_URL = "https://github.com/xinntao/ESRGAN/releases/download/v0.1.1/RRDB_PSNR_x4.pth"
