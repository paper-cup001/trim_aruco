"""
Render.com用に下記のようにしている。
# WSGI サーバーの設定
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
"""

import cv2
import numpy as np
import base64
import logging
from flask import Flask, render_template, request, jsonify
import imghdr
import threading
import gc  # ガベージコレクションを明示的に使用

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

lock = threading.Lock()  # スレッドセーフネスのためにロックを導入

class ArucoDetector:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params)

    def detect_markers(self, image):
        with lock:
            corners, ids, rejectedImgPoints = self.detector.detectMarkers(image)
        return corners, ids, rejectedImgPoints

detector = ArucoDetector()

# 画像ファイルの上限: 5MB = 5 * 1024 * 1024 バイト
MAX_IMAGE_SIZE = 5 * 1024 * 1024

# 出力画像の縦・横の最大ピクセル
max_dim = 1080    # メルカリ仕様

############
# 画像ファイルの確認
############

def validate_image_file(image_file, ip_address):
    """
    フロントエンドから送信されたファイルが画像ファイルで、5MB以下であることを確認する。
    """
    image_size = len(image_file.read())
    logging.info(f'{ip_address} image size: {image_size}')  # ファイルサイズのログ

    if image_size > MAX_IMAGE_SIZE:
        logging.error(f'{ip_address} Image size exceeds the 5MB limit.')
        return False, 'Image size is too large. Maximum allowed size is _MB.'

    image_file.seek(0)

    if not imghdr.what(None, h=image_file.read()):
        logging.error(f'{ip_address} Uploaded file is not a valid image.')
        return False, 'Uploaded file is not a valid image.'
    
    image_file.seek(0)
    return True, None

##############
# 画像データをバイナリで読み込み、OpenCV形式に変換する関数
##############

def read_image(image_file):
    """
    画像データをバイナリで読み込む。
    """
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

##########
# Arucoマーカーの頂点座標を決定
##########

def get_top(corners, side):
    """
    arucoマーカーの右上or左上を検出するための関数。
    右上と左上の頂点座標を選択するための関数。
    """
    top_points = sorted(corners, key=lambda x: x[1])[:2]
    if side == 'left':
        return max(top_points, key=lambda x: x[0])
    else:
        return min(top_points, key=lambda x: x[0])

#########
# Arucoマーカーのエラーハンドリング
##########

def filter_valid_markers(corners, ids):
    # ID が 0,1,2 以外のものは無視する
    valid_ids = [0, 1, 2]
    d_aruco = {}
    
    for i in range(len(ids)):
        if ids[i][0] in valid_ids:
            logging.info(f'{ids[i][0]} のマーカーは有効です')
            d_aruco[ids[i][0]] = corners[i]
        else:
            logging.warning(f'無視するマーカー: ID={ids[i][0]}, オリジナルの座標 {corners[i]}')
    
    return d_aruco

#########
# 画像を正方形にする関数
##########

def make_square(image, min_x, max_x, min_y, max_y):
    """
    トリミングした画像を正方形に整形し、白色で埋める。
    cv2.copyMakeBorder() で余白を追加している。
    """
    height = max_y - min_y
    width = max_x - min_x

    # 長い辺の長さに基づいて正方形を作成
    square_side = max(height, width)
    
    if height > width:
        # 縦が長い場合、横に白色で埋める
        pad_width = (square_side - width) // 2
        image_square = cv2.copyMakeBorder(image[min_y:max_y, min_x:max_x], 0, 0, pad_width, square_side - width - pad_width, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        # 横が長い場合、縦に白色で埋める
        pad_height = (square_side - height) // 2
        image_square = cv2.copyMakeBorder(image[min_y:max_y, min_x:max_x], pad_height, square_side - height - pad_height, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    return image_square

#########
# ここからメイン
#########

@app.route('/')
def index():
    print('------------------')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    print('------------------')
    
    ip_address = request.remote_addr
    logging.info(f'{ip_address} から受信しました')

    if 'image' not in request.files:
        logging.error(f'{ip_address} No image file found in request.')
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']

    valid, error_message = validate_image_file(image_file, ip_address)
    if not valid:
        return jsonify({'error': error_message}), 400

    try:
        image = read_image(image_file)
        image_file.close()
        
        x_offset = int(request.form.get('x_offset', 0))
        y_offset = int(request.form.get('y_offset', 0)) * (-1)  # 常にマイナス
        mode = request.form.get('mode', 'outline')
        logging.info(f'{ip_address} Mode: {mode}, X Offset: {x_offset}, Y Offset: {y_offset}')   # ユーザーの指示のログ

        corners, ids, rejectedImgPoints = detector.detect_markers(image)
        
        if ids is not None:
            d_aruco = filter_valid_markers(corners, ids)

            logging.info(f'{ip_address} 認識したマーカーの数: {len(d_aruco)}')
            
            if len(d_aruco) != 3:
                error_message = str('Error. Please upload exactly 3 ArUco markers (ID=0, ID=1, ID=2). N_marker = {}'.format(len(d_aruco)))
                logging.error(f'{ip_address} Error. マーカーが{len(d_aruco)}つしか認識できません。3つ必要です') 
                return jsonify({'error': error_message}), 400
        else:
            logging.error(f'{ip_address} Error. There is no ArUco marker')
            return jsonify({'error': 'Error. There is no ArUco marker'}), 400  

        center_x = image.shape[1] / 2

        try:
            marker_0_x = d_aruco[0][0][0][0]
        except Exception as e:
            error_message = str(f'Error. ID=0 のマーカーが見つかりません: {str(e)}')
            logging.error(error_message)
            
            return jsonify({'error': 'ID=0のマーカーが見つかりません'}), 500

        if marker_0_x > center_x:
            side = 'right'
        else:
            side = 'left'
            x_offset = -x_offset # 左向きならx_offset値は逆。

        logging.info(f'{ip_address} The marker(ID=0) is on the {side}')   # マーカーの向き

        extracted_points = {}

        for marker_id, corners in d_aruco.items():
            extracted_points[marker_id] = get_top(corners[0], side)

        x_coords = [int(point[0]) for point in extracted_points.values()]
        y_coords = [int(point[1]) for point in extracted_points.values()]

        min_x, max_x = min(x_coords) + x_offset, max(x_coords) + x_offset
        min_y, max_y = min(y_coords) + y_offset, max(y_coords) + y_offset

        if min_x == max_x or min_y == max_y :
            error_message = str('Error. マーカーの位置関係が想定外です')
            
            logging.error(f'{ip_address} マーカーの位置関係が想定外です。min_x :{min_x}, max_x {max_x}, min_y {min_y}, max_y {max_y}')
            
            return jsonify({'error': error_message}), 400

        elif mode == "outline":
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 10)
            resized_image = resize_image_if_needed(image, ip_address)        
            return img_base64(resized_image)

        else:
            square_image = make_square(image, min_x, max_x, min_y, max_y)
            logging.info(f'{ip_address} 解像度{max_dim}の正方形に変更 ')

            resized_image = resize_image_if_needed(square_image, ip_address)
            return img_base64(resized_image)

    finally:
        # クリア
        del image
        gc.collect()  # 明示的なガベージコレクション

def img_base64(image):
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    del buffer
    return jsonify({'image': 'data:image/png;base64,' + img_base64})

def resize_image_if_needed(img, ip_address):
    height, width = img.shape[:2]
    
    if max(height, width) > max_dim:
        scale_ratio = max_dim / max(height, width)
        img = cv2.resize(img, (int(width * scale_ratio), int(height * scale_ratio)))
        
        logging.info(f'{ip_address} フロントエンドに返す解像度を{max_dim}に変更 ')

    return img

# WSGI サーバーの設定
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
