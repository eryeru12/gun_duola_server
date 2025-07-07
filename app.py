from gevent import monkey
monkey.patch_all()

import os
import cv2
import numpy as np
import magic
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter
from mtcnn import MTCNN
import mediapipe as mp
#import dlib
from rembg import remove
from esrgan import ESRGAN

from srcnn_esrgan import SRCNN_ESRGAN
#from srcnn_esrgan import adjust_contrast_brightness, sharpen_image, super_resolve_image_srcnn, super_resolve_image_esrgan, preprocess_image, load_srcnn_model, load_esrgan_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 证件照标准尺寸(单位:mm 转换为像素 300dpi)
SIZE_PRESETS = {
    '1': (295, 413),  # 1寸: 25×35mm -> 295×413px
    '2': (413, 579),  # 2寸: 35×49mm -> 413×579px 
    '4': (600, 800),  # 4寸: 60×80mm -> 600×800px
    '6': (900, 1200)  # 6寸: 90×120mm -> 900×1200px
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image_pipeline(img, bg_color):
    
    # 2. Use rembg to remove background
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    output = remove(pil_img)
    
    # Convert back to OpenCV format for SRCNN
    output_np = np.array(output.convert('RGB'))
    output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    adjusted_img = SRCNN_ESRGAN.adjust_contrast_brightness(output_np, factor=1.5, brightness=30)

    sharpened_img = SRCNN_ESRGAN.sharpen_image(adjusted_img, sigma=1.0)

    #preprocessed_image_srcnn = preprocess_image(adjusted_img)

    preprocessed_image_esrgan = SRCNN_ESRGAN.preprocess_image(sharpened_img)

    #srcnn_model = load_srcnn_model('srcnn_weights.h5')  # Load SRCNN model
    esrgan_model = SRCNN_ESRGAN.load_esrgan_model('models/esrgan_weights.pth')  # Load ESRGAN model

    #super_resolve_image_srcnn = super_resolve_image_srcnn(srcnn_model, preprocessed_image_srcnn, scale=3)

    super_resolve_image_esrgan = super_resolve_image_esrgan(esrgan_model, preprocessed_image_esrgan, scale=4)

    output = Image.fromarray(super_resolve_image_esrgan)
    
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
    edge_region = np.where((dist > 3) & (dist < 10), 1.0, 0.0).astype(np.float32)
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
    
    return result

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/image/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # 获取参数
    bg_color = request.form.get('bg_color', 'white')
    size = request.form.get('size', '1')
    
    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # 保存临时文件
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    
    try:
        # 读取图片
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Process image through pipeline
        processed_img = process_image_pipeline(img, bg_color)
        
        # 按证件照尺寸裁剪人物(居中)
        target_width, target_height = SIZE_PRESETS.get(size, SIZE_PRESETS['1'])
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
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
        cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        # 返回Base64编码的最高质量图片
        _, buffer = cv2.imencode('.png', processed_img)  # 使用无损PNG格式
        img_str = buffer.tobytes().hex()
        
        return jsonify({
            'processed_image': f'data:image/png;base64,{img_str}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)
