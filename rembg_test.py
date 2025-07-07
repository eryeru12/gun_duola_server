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
    
    # 2. Use rembg to remove background
    #pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #pil_img = Image.fromarray(img)  # Assuming img is already in RGB format

    output = remove(img, session=rembg.new_session('u2net_human_seg', './models'))
    #output = remove(pil_img)
    
    output=ESRGAN(model_path='models/esrgan_weights.pth').predict(output)  # Load ESRGAN model
    
    # 3. Create new background
    if bg_color == 'white':
        new_bg = Image.new('RGB', output.size, (255, 255, 255))
    elif bg_color == 'blue':
        new_bg = Image.new('RGB', output.size, (0, 0, 255))
    else:  # red
        new_bg = Image.new('RGB', output.size, (255, 0, 0))
    
    # 优化边缘处理(保持清晰度)
    #mask = output.split()[-1]
    
    # 适度羽化(降低模糊半径)
    #base_radius = max(2, min(5, int(output.width / 200)))
    #mask = mask.filter(ImageFilter.GaussianBlur(radius=base_radius))
    #output.putalpha(mask)
    
    # 智能边缘处理方案
    # 1. 获取精确alpha通道
    mask = np.array(output.split()[-1])
    
    # 2. 仅处理真实边缘区域(避免影响面部和毛发)
    # 计算距离变换找到真正边缘
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    # 只处理距离边缘3-10像素的区域
    edge_region = np.where((dist > 0) & (dist < 5), 1.0, 0.0).astype(np.float32)
    edge_region = cv2.GaussianBlur(edge_region, (0,0), sigmaX=2)
    
    # 3. 颜色混合(仅在边缘区域)
    output_np = np.array(output.convert('RGB'))
    bg_color_np = np.array(new_bg)
    
    for c in range(3):
        blended = (output_np[:,:,c] * (1.0 - edge_region) + 
                  bg_color_np[:,:,c] * edge_region)
        output_np[:,:,c] = np.where(edge_region > 0, blended, output_np[:,:,c]).astype(np.uint8)
    
    # 4. 最终合成
    output = Image.fromarray(output_np).convert('RGBA')
    output.putalpha(Image.fromarray(mask))
    new_bg.paste(output, (0, 0), output)
    result = cv2.cvtColor(np.array(new_bg), cv2.COLOR_RGB2BGR)
       
   
    # 5. 轻微降噪
    result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 5, 15)
    
    # 细节增强
    #result = cv2.detailEnhance(result, sigma_s=15, sigma_r=0.2)
    
    
    
    # 4. dlib quality check
    #detector = dlib.get_frontal_face_detector()
    #faces = detector(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    #if not faces:
    #    raise ValueError("Quality check failed - no faces detected")

            # 按证件照尺寸裁剪人物(居中)
    processed_img = result.copy()
    target_width, target_height = SIZE_PRESETS.get(2, SIZE_PRESETS['1'])
    h, w = processed_img.shape[:2]
        
    # 计算缩放比例并保持宽高比
    scale = min(target_width/w, target_height/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(processed_img, (new_w, new_h))
        
    # 创建目标尺寸画布并居中放置
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if bg_color == 'white':
        canvas.fill(255)
    elif bg_color == 'blue':
        canvas[:,:,2] = 255  # 蓝色背景
    else:  # red
        canvas[:,:,0] = 255  # 红色背景
            
    # 计算居中位置
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    processed_img = canvas
        
    # 保存处理后的图片(最高质量)
    output_path = os.path.join('uploads', f'processed_white.png')
    # 返回Base64编码的最高质量图片
    _, buffer = cv2.imencode('.png', processed_img)  # 使用无损PNG格式
    cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    return output_path


temp_path = os.path.join('uploads', 'zhaopian.png')
#img = cv2.imread(temp_path)
img = Image.open(temp_path).convert('RGB')  # Ensure image is in RGB format
   
bg_color = 'white'  # Example background color, can be 'white', 'blue', or 'red'
        
        # Process image through pipeline
processed_img = process_image_pipeline(img, bg_color)