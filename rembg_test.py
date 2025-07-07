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
    # rembg返回的是RGBA格式的numpy数组
    if isinstance(output, np.ndarray):
        # 保留RGBA格式
        output_np = output
        mask = output_np[:,:,3]  # 提取alpha通道
    else:
        output_np = np.array(output.convert('RGBA'))
        mask = np.array(output.split()[-1])  # 提取alpha通道
    
    # 转换为BGR用于后续处理，确保保留3个通道
    if output_np.shape[2] == 4:  # RGBA
        output_rgb = cv2.cvtColor(output_np, cv2.COLOR_RGBA2BGR)
    else:  # 已经是RGB
        output_rgb = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    print(f"Output RGB shape: {output_rgb.shape}")  # 调试输出
    
    # 3. 使用ESRGAN进行超分辨率重建
    esrgan_model_path = 'models/esrgan_weights.pth'
    if os.path.exists(esrgan_model_path):
        # 图像预处理 - 转换为float32并归一化
        output_rgb = output_rgb.astype(np.float32) / 255.0
        
        esrgan = ESRGAN(model_path=esrgan_model_path)
        
        try:
                # 限制最大放大尺寸
            max_size = 2000
            h, w = output_rgb.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                output_rgb = cv2.resize(output_rgb, (int(w*scale), int(h*scale)))
                
            output_rgb = esrgan.predict(output_rgb)
                
                # 后处理 - 锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            output_rgb = cv2.filter2D(output_rgb, -1, kernel)
                
            print("ESRGAN超分辨率重建完成")
        except Exception as e:
            print(f"ESRGAN预测失败: {str(e)}")
    else:
        print(f"警告: ESRGAN模型文件 {esrgan_model_path} 不存在，跳过超分辨率步骤")
        
    # 确保图像值在0-255范围
    output_rgb = np.clip(output_rgb * 255, 0, 255).astype(np.uint8)
    
    # 3. Create new background (使用PIL.Image)
    width, height = output_np.shape[1], output_np.shape[0]
    if bg_color == 'white':
        new_bg = Image.new('RGB', (width, height), (255, 255, 255))
    elif bg_color == 'blue':
        new_bg = Image.new('RGB', (width, height), (0, 0, 255))
    else:  # red
        new_bg = Image.new('RGB', (width, height), (255, 0, 0))
    
    # 5. 智能边缘处理
    # 确保mask是单通道8位图像
    mask = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    edge_region = np.where((dist > 0) & (dist < 5), 1.0, 0.0).astype(np.float32)
    edge_region = cv2.GaussianBlur(edge_region, (0,0), sigmaX=2)
    
    # 使用numpy进行混合
    new_bg_np = np.array(new_bg)
    for c in range(3):
        blended = (output_rgb[:,:,c] * (1.0 - edge_region) + 
                  new_bg_np[:,:,c] * edge_region)
        output_rgb[:,:,c] = np.where(edge_region > 0, blended, output_rgb[:,:,c]).astype(np.uint8)
    
    # 4. 最终合成
    print(f"Before final conversion - output_rgb shape: {output_rgb.shape}")  # 调试输出
    output = Image.fromarray(output_rgb).convert('RGBA')
    output.putalpha(Image.fromarray(mask))
    new_bg.paste(output, (0, 0), output)
    
    # 确保最终结果是3通道BGR
    result = np.array(new_bg)
    if result.ndim == 2 or result.shape[2] == 1:  # 灰度图
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    elif result.shape[2] == 4:  # RGBA
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    print(f"Final result shape: {result.shape}")  # 调试输出
    
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
