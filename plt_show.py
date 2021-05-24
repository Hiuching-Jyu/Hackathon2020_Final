import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from keras.preprocessing import image
import numpy as np
import keras
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
import streamlit as st
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras import backend as K
import ch

# ############TEST#########

ch.set_ch()
base_dir = 'D:/Python_document/Hackathon2020/TEMP/chest_xray kaggle/chest_xray'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=163,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=163,
        class_mode='categorical')

base_model = InceptionV3(weights='imagenet', include_top=False)
#
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)

for layer in base_model.layers:
    layer.trainable = False  # Freeze the layers not to train

final_model = keras.models.Model(inputs=base_model.inputs, outputs=predictions)

final_model.compile(loss="categorical_crossentropy",  # another term for log loss
                    optimizer="adam",
                    metrics=["accuracy"])


history = final_model.fit_generator(
    train_generator,
    steps_per_epoch=1,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
# ###########TEST############
# 训练数据精度和损失建模

train_dir = '.chest_xray/train/'
val_dir = 'chest_xray/test/'
test_dir = 'chest_xray/val/'

# img = mping.imread(train_dir + 'NORMAL/IM-0115-0001.jpeg')
# imgplot = plt.imshow(img)
# plt.show()


final_model.save('model1.h5')
K.clear_session()
new_model = load_model("model1.h5")


def data_visualization(image_path):
    # "/content/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg" #@param {type:"string"}

    _img = image.load_img(image_path, target_size=(150, 150))

    _x = image.img_to_array(_img)
    _x = np.expand_dims(_x, axis=0)
    _x = preprocess_input(_x)
    _y = new_model.predict(_x)

    predict = "X光胸片正常，判定为非肺炎患者" if _y.argmax(axis=-1) == 0 else "X光胸片异常，判定为肺炎患者"
    actual = "实际表现为正常" if "NORMAL" in image_path else "实际表现为异常"

    _img = cv2.imread(image_path)
    # title_text = ("%s%s%s%s%s" % ("Original Label: ", actual, "\n", "Prediction: ", predict))
    # plt.title(title_text)
    # _imgplot = plt.imshow(_img)
    # plt.show()

    # data visualization
    if 223 % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(9, 9), facecolor='#F0F0F0')
    plt.tight_layout()
    ax1 = plt.subplot(221)
    ax1.set_title('模型准确率', fontsize=14)
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.set_xlabel('训练次数', fontsize=12)
    ax1.set_facecolor('#F8F8F8')
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.legend(['训练集', '验证集'])

    ax2 = plt.subplot(222)
    ax2.set_title('模型损失率', fontsize=14)
    ax2.set_ylabel('损失率', fontsize=12)
    ax2.set_xlabel('训练次数', fontsize=12)
    ax2.set_facecolor('#F8F8F8')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.legend(['训练集', '验证集'])

    ax3 = plt.subplot(223)
    ax3.set_title('')
    title_text = ("%s%s%s%s%s" % ("实际情况 ", actual, "\n", "预测情况 ", predict))
    plt.title(title_text)
    _imgplot = plt.imshow(_img)
    st.pyplot()


data_visualization('D:/Python_document/Hackathon2020/TEMP/chest_xray kaggle/val/NORMAL/NORMAL2-IM-1427-0001.jpeg',)





