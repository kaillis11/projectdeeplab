import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# 경로 설정
MODEL_PATH = './deeplab_model.h5'
TEST_IMAGE_DIR = './test_images'
OUTPUT_DIR = './test_results'

# 결과 저장할 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_image(image_path):
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert('RGB')
    img = img.resize((624, 288))  # 학습할 때 사용한 크기로 조정
    img_array = np.array(img) / 255.0  # 정규화
    return np.expand_dims(img_array, axis=0)  # 배치 차원 추가

def predict_mask(model, image_path):
    # 이미지 로드 및 예측
    input_image = load_and_preprocess_image(image_path)
    predicted_mask = model.predict(input_image)
    return predicted_mask[0, :, :, 0]  # 첫 번째 배치의 마스크 반환

def visualize_results(image_path, predicted_mask, save_path):
    # 결과 시각화
    original_image = Image.open(image_path).convert('RGB')
    original_image = original_image.resize((624, 288))
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Overlay')
    overlay = np.array(original_image)
    overlay[:, :, 1] = np.where(predicted_mask > 0.5, 
                               overlay[:, :, 1] * 0.7 + 255 * 0.3, 
                               overlay[:, :, 1])
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 저장된 모델 로드
    print("모델 로딩 중...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 테스트 이미지 처리
    print("테스트 이미지 처리 중...")
    for image_name in os.listdir(TEST_IMAGE_DIR):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(TEST_IMAGE_DIR, image_name)
            
            # 결과 파일 경로
            result_path = os.path.join(OUTPUT_DIR, f'result_{image_name}')
            mask_path = os.path.join(OUTPUT_DIR, f'mask_{image_name}')
            
            # 예측 및 시각화
            predicted_mask = predict_mask(model, image_path)
            visualize_results(image_path, predicted_mask, result_path)
            
            # 마스크 저장
            mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
            mask_image.save(mask_path)
            
            print(f"처리 완료: {image_name}")
    
    print("모든 테스트 이미지 처리 완료!")

if __name__ == '__main__':
    main()
