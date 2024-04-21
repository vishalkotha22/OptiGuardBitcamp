import tensorflow as tf
import keras
# import keras.layers as layers
# import keras.models as datasets 
# import keras.models as models
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import cv2
#Check if these imports work
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

print('monkey')

#train = cv2.imread("Saketh\\*")

ipt = (64, 64, 3)
num_classes = 4 #amount of people

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=ipt),
    layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#----------------------------
# Load and preprocess the training data
"""
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    'Saketh',
    target_size=(32, 32),
    batch_size=32,
    class_mode  ='categorical')


print(train_generator)
exit()
train_generator = np.array([])

# Fit the model
# model.fit(
#    train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     epochs=10)
"""

train_dir = 'train'

names = ["Vishal", "Hemang", "Saketh", "Kevin"]

# Define data generator for training
train_datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Same as input shape
    batch_size=20,
    class_mode='categorical',  # Multi-class classification
    shuffle=True  # Shuffle data for better training
)

# Define data generator for validation
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=20,
    class_mode='categorical',
    subset='validation'  # Use the validation subset
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=10,  # Adjust according to the size of your dataset
    epochs=30,  # Adjust as needed
    validation_data=validation_generator,
    validation_steps = validation_generator.samples // validation_generator.batch_size
)

model.save('finalmodel.h5')
print(history)

