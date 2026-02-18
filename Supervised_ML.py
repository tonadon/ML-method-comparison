import pandas as pd  
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import time

cur_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(cur_dir, 'CatsDogsDataset', 'train')
test_dir  = os.path.join(cur_dir, 'CatsDogsDataset', 'test')


def load_images_from_folder(dir, imgs, fns):
    for filename in os.listdir(dir):
        img_path = os.path.join(dir, filename)
        if os.path.isdir(img_path):
            load_images_from_folder(img_path, imgs, fns)
        else:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                img = np.array(img)
                img = img.flatten()
                images.append(img)
                filenames.append(filename)


# Load training and testing data
images =[]
filenames =[]
load_images_from_folder(train_dir, images, filenames)

X_train = images
y_train = [fn.split('_')[0] for fn in filenames]

images =[]
filenames =[]
load_images_from_folder(test_dir, images, filenames)

X_test = images
y_test = [fn.split('_')[0] for fn in filenames]

model = LogisticRegression()
model.fit(X_train, y_train)
t = time.time()
pred = model.predict(X_test)
print(time.time() - t)

pred_proba = model.predict_proba(X_test) 
precision = precision_score(y_test, pred, pos_label='dog')
recall = recall_score(y_test, pred, pos_label='dog')
f1 = f1_score(y_test, pred, pos_label='dog')
accuracy = accuracy_score(y_test, pred)

metrics = [precision, recall, f1, accuracy]
metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']

plt.figure(figsize=(10, 6))
plt.bar(metric_names, metrics, color=['blue', 'orange', 'green', 'red'])
plt.ylabel('Score')
plt.title('Model Evaluation Metrics')
plt.ylim(0, max(metrics) + 0.1)
plt.grid(axis='y')

print(accuracy)
print(precision)
print(recall)
print(f1)

plt.show()