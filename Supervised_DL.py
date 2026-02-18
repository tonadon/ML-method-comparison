
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt  
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras import datasets, layers, models, losses, backend
from tensorflow.keras.applications import MobileNetV2  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import time

def precision(y_true, y_pred):
    y_true = backend.cast(y_true, backend.floatx())  # 转换为 float32
    y_pred = backend.cast(y_pred, backend.floatx())  # 转换为 float32
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + backend.epsilon())

def recall(y_true, y_pred):
    y_true = backend.cast(y_true, backend.floatx())  # 转换为 float32
    y_pred = backend.cast(y_pred, backend.floatx())  # 转换为 float32
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + backend.epsilon())

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + backend.epsilon())

cur_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(cur_dir, 'CatsDogsDataset', 'train')
test_dir  = os.path.join(cur_dir, 'CatsDogsDataset', 'test')


def load_data(data_dir, target_size=(256, 256)):
    data = []
    labels = []
    
    # 遍历每个类别文件夹
    for label, class_name in enumerate(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        
        if os.path.isdir(class_dir):  # 确保是文件夹
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                # 加载图像并调整大小
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                data.append(img_array)
                labels.append(label)

    return np.array(data), np.array(labels)

train_data, train_labels = load_data(train_dir)
test_data, test_labels = load_data(test_dir)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /= 255.0
test_data /= 255.0

print(f'Train data shape: {train_data.shape}')
print(f'Train labels shape: {train_labels.shape}')
print(f'Test data shape: {test_data.shape}')
print(f'Test labels shape: {test_labels.shape}')


model = models.Sequential()
model.add(layers.Input(shape=(256, 256, 3)))
model.add(layers.Conv2D(64, (4, 4), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(128, (4, 4), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(256, (4, 4), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Set up early stopping and model checkpoint to prevent overfitting  
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model  
# history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels), callbacks=[early_stopping, checkpoint])

t = time.time()
loss, accuracy = model.evaluate(test_data, test_labels)
print(time.time() - t)

y_pred_prob = model.predict(test_data)
y_pred_classes = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
f1 = f1_score(test_labels, y_pred_classes)
recall = recall_score(test_labels, y_pred_classes)
precision = precision_score(test_labels, y_pred_classes)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Step 5: Plot Training History  
plt.figure(figsize=(12, 4))

# Accuracy plot  
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot  
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


