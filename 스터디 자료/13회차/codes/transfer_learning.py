import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import random


# In[]

input_width = 224
input_height = 224
input_channel = 3

num_class = 4

BATCH_SIZE = 16

img_dir = './data/'
labels = os.listdir(img_dir)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split = 0.1)

train_generator = datagen.flow_from_directory(
    img_dir,
    target_size=(input_width, input_height),
    batch_size=BATCH_SIZE,
    subset='training')

val_generator = datagen.flow_from_directory(
    img_dir,
    target_size=(input_width, input_height),
    batch_size=BATCH_SIZE,
    subset='validation')

print(train_generator.class_indices)


# In[]
base_model = tf.keras.applications.VGG16(input_shape = (input_width, input_height, input_channel),
                                        include_top = False,
                                        weights = None)


base_model.summary()


# In[]

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_class, activation='softmax')
])

model.summary()


# In[]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
              loss='CategoricalCrossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs = 5,
                    validation_data=val_generator)


# Epoch 1/5
#  2/27 [=>............................] - ETA: 4s - loss: 1.4161 - accuracy: 0.3125WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.1032s vs `on_train_batch_end` time: 0.2725s). Check your callbacks.
# 27/27 [==============================] - 15s 563ms/step - loss: 0.4176 - accuracy: 0.8447 - val_loss: 0.1802 - val_accuracy: 0.954
# Epoch 2/5
# 27/27 [==============================] - 11s 416ms/step - loss: 0.0446 - accuracy: 0.9812 - val_loss: 0.2144 - val_accuracy: 0.954
# Epoch 3/5
# 27/27 [==============================] - 11s 416ms/step - loss: 0.0231 - accuracy: 0.9906 - val_loss: 0.0544 - val_accuracy: 0.977
# Epoch 4/5
# 27/27 [==============================] - 11s 417ms/step - loss: 0.0639 - accuracy: 0.9812 - val_loss: 0.0356 - val_accuracy: 1.000
# Epoch 5/5
# 27/27 [==============================] - 11s 416ms/step - loss: 0.0079 - accuracy: 0.9976 - val_loss: 0.1725 - val_accuracy: 0.954


# In[]
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# In[]

import time

labels

img_list = os.listdir('./test_imgs/')

inference_time_list = []
acc = []

for img_file in img_list:
    img = Image.open("./test_imgs/"+img_file).resize([input_width, input_height])

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    strt_time = time.time()
    predicted = model.predict(img_array)
    end_time = time.time()

    inference_time_list.append(end_time-strt_time)

    res = np.argmax(predicted)

    if labels[res] == img_file.split("_")[0]:
        acc.append(1)
    else : acc.append(0)

print(np.mean(inference_time_list))
print(np.mean(acc))
