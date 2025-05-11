import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Dataset path
dataset_path = "Dataset"

# Show number of images per category
categories = os.listdir(dataset_path)
for category in categories:
    path = os.path.join(dataset_path, category)
    print(f"{category}: {len(os.listdir(path))} images")

# Data Augmentation & Preprocessing (Train & Validation)
datagen_train_val = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.3  # 30% for validation + test
)

# Data Preprocessing (Test)
datagen_test = ImageDataGenerator(rescale=1.0/255)

# Parameters
batch_size = 16
img_size = (224, 224)

# Load training data (70%)
train_generator = datagen_train_val.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data (30%)
val_generator = datagen_train_val.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Further split validation into 15% validation & 15% test
val_split_ratio = 0.5
num_val_samples = int(len(val_generator.filenames) * val_split_ratio)

# Load test data (from same validation split but reduced)
test_generator = datagen_test.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
test_generator.filenames = test_generator.filenames[:num_val_samples]
test_generator.samples = num_val_samples

# Number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Load VGG16 (without top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# Define final model
model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=epochs
)

# Save model
model.save("/content/vgg16_finetuned.h5")

# Plot training vs validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('VGG16 Fine-Tuning Accuracy')
plt.legend()
plt.grid(True)
plt.show()
