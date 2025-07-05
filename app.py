from gevent import monkey
monkey.patch_all()

import os
import cv2
import numpy as np
import magic
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter

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

def change_background(img, bg_color):
    """更换图片背景色"""
    # 转换到LAB色彩空间 - 对亮度变化更敏感
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # 自动检测背景色 (假设背景是均匀的)
    bg_pixels = np.concatenate([
        lab[0:5, 0:5],       # 左上角
        lab[0:5, -5:],       # 右上角 
        lab[-5:, 0:5],       # 左下角
        lab[-5:, -5:]        # 右下角
    ])
    bg_mean = np.mean(bg_pixels, axis=0)
    
    # 根据背景亮度动态调整阈值
    threshold = 25 if bg_mean[0] > 150 else 15  # 亮背景用大阈值
    
    # 创建背景蒙版
    mask = cv2.inRange(lab, 
                      bg_mean - threshold,
                      bg_mean + threshold)
    
    # 改进的形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 边缘优化
    mask = cv2.GaussianBlur(mask, (21,21), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 使用GrabCut优化困难案例
    if np.mean(mask) > 0.4:  # 如果背景占比较大
        bgd_model = np.zeros((1,65), np.float64)
        fgd_model = np.zeros((1,65), np.float64)
        mask, _, _ = cv2.grabCut(img, mask, None, bgd_model, fgd_model, 
                               5, cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')*255
    
    # 创建新背景
    if bg_color == 'white':
        new_bg = np.ones_like(img) * 255
    elif bg_color == 'blue':
        new_bg = np.zeros_like(img)
        new_bg[:,:,0] = 255  # OpenCV使用BGR格式
    else:  # red
        new_bg = np.zeros_like(img)
        new_bg[:,:,2] = 255
    
    # 合并前景和背景
    result = np.where(mask[:,:,np.newaxis]==0, img, new_bg)
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
        
        # 更换背景
        processed_img = change_background(img, bg_color)
        
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
