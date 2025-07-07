from PIL import Image
from rembg import remove
import cv2
from rembg import new_session
import rembg
from srccn_esrgan import SRCNN_ESRGAN
import numpy as np
import os
from esrgan import ESRGAN

# 证件照标准尺寸(单位:mm 转换为像素 300dpi)
SIZE_PRESETS = {
    '1': (295, 413),  # 1寸: 25×35mm -> 295×413px
    '2': (413, 579),  # 2寸: 35×49mm -> 413×579px 
    '4': (600, 800),  # 4寸: 60×80mm -> 600×800px
    '6': (900, 1200)  # 6寸: 90×120mm -> 900×1200px
}

def process_image_pipeline(img, bg_color):
    # 1. 转换为OpenCV格式
    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # 2. 使用rembg去除背景
    output = remove(img, session=rembg.new_session('u2net_human_seg', './models'))
    output_np = np.array(output.convert('RGB'))
    output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    # 3. 使用ESRGAN进行超分辨率重建
    esrgan = ESRGAN(model_path='models/esrgan_weights.pth')
    if esrgan.model is not None:
        output_np = esrgan.predict(output_np)
    
    # 4. 创建新背景
    if bg_color == 'white':
        new_bg = np.full_like(output_np, 255)
    elif bg_color == 'blue':
        new_bg = np.full_like(output_np, (255, 0, 0))
    else:  # red
        new_bg = np.full_like(output_np, (0, 0, 255))
    
    # 5. 智能边缘处理
    mask = np.array(output.split()[-1])
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    edge_region = np.where((dist > 0) & (dist < 5), 1.0, 0.0).astype(np.float32)
    edge_region = cv2.GaussianBlur(edge_region, (0,0), sigmaX=2)
    
    for c in range(3):
        blended = (output_np[:,:,c] * (1.0 - edge_region) + 
                  new_bg[:,:,c] * edge_region)
        output_np[:,:,c] = np.where(edge_region > 0, blended, output_np[:,:,c]).astype(np.uint8)
    
    # 6. 最终合成
    result = output_np
    
    # 7. 后处理
    result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 5, 15)
    
    # 8. 按证件照尺寸裁剪
    target_width, target_height = SIZE_PRESETS.get('2', SIZE_PRESETS['1'])
    h, w = result.shape[:2]
    scale = min(target_width/w, target_height/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(result, (new_w, new_h))
    
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if bg_color == 'white':
        canvas.fill(255)
    elif bg_color == 'blue':
        canvas[:,:,2] = 255
    else:
        canvas[:,:,0] = 255
    
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # 9. 保存结果
    output_path = os.path.join('uploads', f'processed_{bg_color}.png')
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    
    return output_path

if __name__ == '__main__':
    temp_path = os.path.join('uploads', 'zhaopian.png')
    img = Image.open(temp_path).convert('RGB')
    bg_color = 'white'
    processed_img = process_image_pipeline(img, bg_color)
