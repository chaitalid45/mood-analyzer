import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import os

# Step 1: Data Preparation
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7 

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,             ## rescaling range 0-255 to 0-1
    rotation_range=20,          ## Rotate image by 20 degree
    width_shift_range=0.2,      ## horizontal shift
    height_shift_range=0.2,     ## vertical shift
    horizontal_flip=True,       
    validation_split=0.2
)

# Traing Data 
train_generator = train_datagen.flow_from_directory(
    "images\train",  # labeled images path for the training image
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation Data
val_generator = train_datagen.flow_from_directory(
    "images\validation",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Step 2: Build Model with Transfer Learning
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Freeze initially do not update the weight.

x = base_model.output
x = GlobalAveragePooling2D()(x)         ## convert the D to 1D
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 3: Train Model
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
]

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=callbacks
)

# Fine-tune: Unfreeze top layers
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Fine-tune last 30
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=callbacks)

## Saving the model

model.save('mood_model.h5')
