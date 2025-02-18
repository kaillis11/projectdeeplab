import tensorflow as tf
import numpy as np
import os

#pip install h5py 필수수

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

# TFRecord 데이터 로드 함수
def parse_tfrecord(example_proto):
    # TFRecord 형식 정의 수정
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # 'image/encoded'에서 'image'로 변경
        'mask': tf.io.FixedLenFeature([], tf.string)   # 'image/segmentation/class/encoded'에서 'mask'로 변경
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # 이미지 디코딩
    image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
    mask = tf.io.decode_jpeg(parsed_features['mask'], channels=1)
    
    # 이미지 전처리
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    
    # 정확한 크기로 조정 (288x624)
    image = tf.image.resize(image, [288, 624])
    mask = tf.image.resize(mask, [288, 624])
    
    return image, mask

# 데이터셋 생성
def create_dataset(tfrecord_path, batch_size=1):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# DeepLab 모델 생성
def create_model():
    input_shape = [288, 624, 3]
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # 입력을 MobileNetV2에 맞게 리사이즈
    x = tf.keras.layers.Resizing(224, 224)(inputs)
    
    # MobileNetV2를 기본 크기로 초기화
    base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], 
                                                 include_top=False,
                                                 weights='imagenet')
    
    # 전체 base_model을 통과
    x = base_model(x)
    
    # 디코더 레이어 정의
    up_stack = [
        upsample(512, 3),
        upsample(256, 3),
        upsample(128, 3),
        upsample(64, 3),
    ]

    # 업샘플링
    for up in up_stack:
        x = up(x)

    # 최종 출력을 원래 이미지 크기로 조정
    x = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same')(x)
    x = tf.keras.layers.Resizing(288, 624)(x)
    
    # 활성화 함수 추가
    outputs = tf.keras.layers.Activation('sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def main():
    # 로그 디렉토리 생성
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 체크포인트 디렉토리 생성
    if not os.path.exists('training_checkpoints'):
        os.makedirs('training_checkpoints')
    
    print("데이터셋 로드 중...")
    # TFRecord 파일 경로 확인 및 수정
    tfrecord_path = os.path.join(os.path.dirname(__file__), "tfrecords", "train.tfrecord")
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord 파일을 찾을 수 없습니다: {tfrecord_path}")
    
    dataset = create_dataset(tfrecord_path, batch_size=1)
    
    print("모델 생성 중...")
    model = create_model()
    
    # Binary Cross Entropy 설정 수정
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  # from_logits를 False로 변경
                 metrics=['accuracy'])
    
    # 체크포인트 설정
    checkpoint_path = "training_checkpoints/cp-{epoch:04d}.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1,
        save_freq=5*1
    )
    
    # TensorBoard 설정
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='logs',
        update_freq='epoch'
    )
    
    print("학습 시작...")
    history = model.fit(
        dataset,
        epochs=50,
        callbacks=[cp_callback, tensorboard_callback]
    )
    
    print("모델 저장 중...")
    try:
        model.save('deeplab_model.h5')  # .h5 확장자 추가
        print("모델 저장 완료!")
    except Exception as e:
        print(f"모델 저장 중 오류 발생: {str(e)}")
        # 대체 저장 방법 시도
        try:
            model.save_weights('deeplab_model_weights.h5')
            print("모델 가중치 저장 완료!")
        except Exception as e:
            print(f"가중치 저장 중 오류 발생: {str(e)}")
    
    print("학습 완료!")

if __name__ == '__main__':
    main()
