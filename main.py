import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# def create_convolutional_model():
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
#
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
#     BatchNormalization(),
#     Conv2D(32, (3, 3), activation='relu', padding='same'),
#     MaxPooling2D((2, 2)),
#     Dropout(0.25),
#
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     Conv2D(64, (3, 3), activation='relu', padding='same'),
#     MaxPooling2D((2, 2)),
#     Dropout(0.25),
#
#     Conv2D(128, (3, 3), activation='relu', padding='same'),
#     BatchNormalization(),
#     Conv2D(128, (3, 3), activation='relu', padding='same'),
#     MaxPooling2D((2, 2)),
#     Dropout(0.25),
#
#     Flatten(),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.5),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax')
# ])
#
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
#
# callbacks = [
#     EarlyStopping(patience=10, restore_best_weights=True),
#     ReduceLROnPlateau(factor=0.5, patience=5)
# ]
#
# st.write("Обучение модели...")
# history = model.fit(
#     x_train, y_train,
#     batch_size=64,
#     epochs=10,
#     validation_data=(x_test, y_test),
#     callbacks=callbacks,
#     verbose=1
# )

# model.save('cifar_conv_model.keras')

def main():
    st.set_page_config(page_title="Классификация изображений", layout="wide")

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .probability-bar {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header"> Классификация изображений </h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(" Загрузка изображения")
        file = st.file_uploader('Выберите файл .jpg или .png', type=['jpg', 'png'],
                                help="Загрузите изображение для классификации")

        if not file:
            st.markdown("---")
            st.info("""
            - О модели:
            - Сверточная нейронная сеть обучена на датасете CIFAR-10
            - 10 категорий объектов
             - Точность модели: ~85%
             """)
        if file:
            image = Image.open(file)
            st.image(image, caption="Ваше изображение", use_container_width=True)



    with col2:
        if file:
            st.subheader(" Результаты классификации")

            with st.spinner(" Анализируем изображение..."):
                # Обработка изображения
                resized = image.resize((32, 32))
                img_array = np.array(resized) / 255
                img_array = img_array.reshape((1, 32, 32, 3))

                # Загрузка модели и предсказание
                model = tf.keras.models.load_model('cifar_conv_model.keras')
                predictions = model.predict(img_array, verbose=0)

                classes = [
                    ' Самолёт', ' Автомобиль', ' Птица', ' Кошка', ' Олень',
                    ' Собака', ' Лягушка', ' Лошадь', ' Корабль', ' Грузовик'
                ]

                # Находим максимальную вероятность
                max_prob_idx = np.argmax(predictions[0])
                max_prob = predictions[0][max_prob_idx]

                # Отображаем основной результат
                st.markdown(f"""
                <div class="result-card">
                    <h3>Результат: {classes[max_prob_idx]}</h3>
                    <h1>{max_prob:.1%}</h1>
                    <p>Уверенность модели в предсказании</p>
                </div>
                """, unsafe_allow_html=True)

                # Детальная информация по классам
                st.subheader("📈 Вероятности по всем категориям:")

                for i, (cls, prob) in enumerate(zip(classes, predictions[0])):
                    progress = int(prob * 100)
                    color = "🟢" if i == max_prob_idx else "⚪"

                    col_a, col_b, col_c = st.columns([2, 6, 2])
                    with col_a:
                        st.write(f"{color} {cls}")
                    with col_b:
                        st.progress(progress, text=f"{prob:.1%}")
                    with col_c:
                        st.write(f"{prob:.1%}")

                # График
                st.subheader("📊 Визуализация вероятностей")
                fig, ax = plt.subplots(figsize=(10, 6))

                colors = ['lightcoral' if i == max_prob_idx else 'lightblue'
                          for i in range(len(classes))]

                bars = ax.barh(classes, predictions[0], color=colors)
                ax.set_xlabel('Вероятность', fontsize=12)
                ax.set_title('Распределение вероятностей по классам', fontsize=14, pad=20)
                ax.set_xlim(0, 1)

                # Добавляем проценты на график
                for bar, prob in zip(bars, predictions[0]):
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{prob:.1%}', ha='left', va='center', fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)



if __name__ == '__main__':
    main()
