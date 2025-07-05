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
    
    # 2. Use rembg with enhanced background removal
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 使用post_process=True和alpha_matting参数增强边缘处理
    output = remove(
        pil_img,
        post_process=True,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10
    )
    
    # 额外边缘清理
    mask = np.array(output.split()[-1])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    output.putalpha(Image.fromarray(mask))
    
    # 3. Create new background
    if bg_color == 'white':
        new_bg = Image.new('RGB', output.size, (255, 255, 255))
    elif bg_color == 'blue':
        new_bg = Image.new('RGB', output.size, (173, 216, 230))  # 浅蓝色
    elif bg_color == 'gray':
        new_bg = Image.new('RGB', output.size, (230, 230, 230))  # 浅灰色
    else:  # red
        new_bg = Image.new('RGB', output.size, (255, 0, 0))
        
    # 人脸美化 - 转换为OpenCV处理
    cv_img = cv2.cvtColor(np.array(output.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # 获取人脸区域
    face_img = cv_img[y:y+h, x:x+w]
    
    # 应用双边滤波美化皮肤
    if face_img.size > 0:
        face_img = cv2.bilateralFilter(face_img, 9, 75, 75)
        cv_img[y:y+h, x:x+w] = face_img
    
    # 优化头型轮廓 - 使用形态学操作
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 转换回PIL格式
    output = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    mask = Image.fromarray(mask)
    output.putalpha(mask)
    
    # Combine with new background
    new_bg.paste(output, (0, 0), output)
    
    # 高级边缘处理
    mask = output.split()[-1]  # 获取alpha通道
    
    # 精确边缘检测
    edge_mask = mask.filter(ImageFilter.FIND_EDGES)
    edge_mask = edge_mask.point(lambda x: 255 if x > 30 else 0)
    
    # 动态羽化处理 (根据图像大小调整半径)
    blur_radius = max(3, min(10, int(output.width / 100)))
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # 边缘锐化处理
    sharp_mask = Image.blend(mask, edge_mask, 0.3)
    final_mask = Image.blend(blurred_mask, sharp_mask, 0.7)
    
    # 应用优化后的边缘
    new_output = output.copy()
    new_output.putalpha(final_mask)
    new_bg.paste(new_output, (0, 0), new_output)
    
    # 最终微调
    result_img = np.array(new_bg)
    result_img = cv2.detailEnhance(result_img, sigma_s=10, sigma_r=0.15)
    
    result = cv2.cvtColor(np.array(new_bg), cv2.COLOR_RGB2BGR)
    
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
        
        # 保存处理后的图片
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
        cv2.imwrite(output_path, processed_img)
        
        # 返回Base64编码的结果图片
        _, buffer = cv2.imencode('.jpg', processed_img)
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
