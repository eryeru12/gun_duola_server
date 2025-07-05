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
    """4-step image processing pipeline"""
    # 1. MTCNN face detection and alignment
    detector = MTCNN()
    faces = detector.detect_faces(img)
    if not faces:
        raise ValueError("No faces detected")
    
    # Get main face (largest bounding box)
    main_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = main_face['box']
    
    # 2. Use rembg to remove background
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    output = remove(pil_img)
    
    # 3. Create new background
    if bg_color == 'white':
        new_bg = Image.new('RGB', output.size, (255, 255, 255))
    elif bg_color == 'blue':
        new_bg = Image.new('RGB', output.size, (0, 0, 255))
    else:  # red
        new_bg = Image.new('RGB', output.size, (255, 0, 0))
    
    # 优化边缘处理(保持清晰度)
    mask = output.split()[-1]
    
    # 适度羽化(降低模糊半径)
    base_radius = max(2, min(5, int(output.width / 200)))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=base_radius))
    output.putalpha(mask)
    
    # 合成背景
    new_bg.paste(output, (0, 0), output)
    result = cv2.cvtColor(np.array(new_bg), cv2.COLOR_RGB2BGR)
    
    # 细节增强
    result = cv2.detailEnhance(result, sigma_s=15, sigma_r=0.2)
    
    # 轻度锐化
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    result = cv2.filter2D(result, -1, kernel)
    
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
        
        # 调整尺寸
        target_size = SIZE_PRESETS.get(size, SIZE_PRESETS['1'])
        processed_img = cv2.resize(processed_img, target_size)
        
        # 保存处理后的图片(高质量)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
        cv2.imwrite(output_path, processed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 返回Base64编码的高质量图片
        _, buffer = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_str = buffer.tobytes().hex()
        
        return jsonify({
            'processed_image': f'data:image/jpeg;base64,{img_str}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)
