import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
import scipy
dataset_path = "archive"
class_names = os.listdir(dataset_path)
num_classes = len(class_names)
input_size = (48, 48)
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.3  # Adjust the split ratio
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Training
num_epochs = 10
history = model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator)