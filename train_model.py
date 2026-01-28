import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# =============================
# CONFIG
# =============================
TRAIN_DIR = r"C:\Users\santo\OneDrive\Desktop\PlantDiseaseDetectionDLProject\train"

CLASSES = [
    'Tomato___Target_Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Late_blight',
    'Tomato___Early_blight',
    'Tomato___healthy'
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 10
NUM_CLASSES = len(CLASSES)

# =============================
# DATA GENERATORS
# =============================

# Training (with augmentation)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Validation (NO augmentation)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# =============================
# CLASS WEIGHTS (SAFE)
# =============================
class_counts = np.bincount(train_gen.classes)

class_weights = {}
max_count = np.max(class_counts)

for i in range(NUM_CLASSES):
    if class_counts[i] == 0:
        class_weights[i] = 1.0
    else:
        class_weights[i] = max_count / class_counts[i]

print("Class Weights:", class_weights)

# =============================
# BASE MODEL
# =============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# =============================
# CUSTOM HEAD
# =============================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# =============================
# COMPILE STAGE 1
# =============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =============================
# CALLBACKS
# =============================
callbacks = [
    ModelCheckpoint(
        "model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        verbose=1
    )
]

# =============================
# TRAIN STAGE 1
# =============================
print("\n🔵 Stage 1: Training classifier head")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=callbacks
)

# =============================
# FINE-TUNING
# =============================
print("\n🟢 Stage 2: Fine-tuning MobileNetV2")

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=callbacks
)

print("\n✅ TRAINING COMPLETE")
print("✅ Model saved as model.h5")
