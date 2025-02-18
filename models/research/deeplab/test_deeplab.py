import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    # 이미지 로드
    img = tf.keras.preprocessing.image.load_img(image_path)
    # 가로로 긴 형태로 리사이즈 (288x624)
    img = img.resize((624, 288))  # PIL Image는 (width, height) 순서
    # 배열로 변환
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # 정규화
    img_array = img_array / 255.0
    # 배치 차원 추가
    img_array = np.expand_dims(img_array, 0)
    return img_array

def visualize_prediction(image, mask):
    plt.figure(figsize=(15, 5))  # 가로로 긴 형태로 시각화
    
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(image[0])
    
    plt.subplot(1, 3, 2)
    plt.title('Predicted Mask')
    plt.imshow(mask[0, :, :, 0], cmap='gray')
    
    # 마스크를 오버레이
    overlay = image[0].copy()
    overlay[mask[0, :, :, 0] > 0.5] = [1, 0, 0]  # 빨간색으로 마스크 표시
    
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    plt.imshow(overlay)
    
    plt.show()

def main():
    # 모델 로드
    print("모델 로딩 중...")
    model = tf.keras.models.load_model('deeplab_model.h5')
    
    # 테스트할 이미지 경로
    test_image_path = r'C:\Users\mwork\Desktop\prpro\project_image\models\research\deeplab\test_image_path\KakaoTalk_20250130_145238023_01.jpg'  # 테스트할 이미지 경로를 지정해주세요
    
    # 이미지 로드 및 전처리
    print("이미지 전처리 중...")
    input_image = load_and_preprocess_image(test_image_path)
    
    # 예측
    print("세그멘테이션 수행 중...")
    prediction = model.predict(input_image)
    
    # 결과 시각화
    print("결과 시각화...")
    visualize_prediction(input_image, prediction)

if __name__ == '__main__':
    main()