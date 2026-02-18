
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import time


# Load images from a local directory


def load_images_from_folder(folder_path, img_size=(224, 224)):
    images = []
    filenames = []
    for foldername in os.listdir(folder_path):
        path = folder_path + '\\' +foldername
        path = os.path.normpath(path)
        for filename in os.listdir(path):
            img_path = path+ '\\' +filename
            img_path = os.path.normpath(img_path)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)  # Resize image
                images.append(img)
                filenames.append(filename)
    return np.array(images), filenames


# Function to load images from a local folder and assign labels based on folder name
def load_images_and_labels_from_folder(folder_path, img_size=(64, 64)):
    images = []
    labels = []
    filenames = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    folder_path = os.path.normpath(folder_path)
    
    # Iterate over the folder, each subfolder represents a class
    for label, subfolder in enumerate(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(valid_extensions):
                    img_path = os.path.join(subfolder_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        images.append(img)
                        labels.append(label)  # Label based on the folder (0 or 1)
                        filenames.append(filename)
                    else:
                        print(f"Failed to load image: {img_path}")

    if not images:
        print("No images were loaded.")
    
    return np.array(images), np.array(labels), filenames


# Preprocess images (flatten them)

 
def preprocess_images(images, img_size=(128, 128)):
    # Resize images to the same shape
    resized_images = np.array([cv2.resize(img, img_size) for img in images])
    
    # Flatten the images (resize to one-dimensional arrays)
    return resized_images.reshape(resized_images.shape[0], -1)  # Flatten each image


# Perform KMeans clustering

 
def perform_kmeans_clustering(features, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

 
# Display clustered images

 
def display_clusters(images, kmeans, filenames, columns_per_row=5):
    labels = kmeans.labels_
    unique_labels = np.unique(labels)
    
    # Determine the number of clusters
    num_clusters = len(unique_labels)
    
    # Find the maximum number of images in any single cluster
    max_images_in_cluster = max([np.sum(labels == label) for label in unique_labels])
    
    # Calculate the number of rows and columns required
    rows = num_clusters
    cols = min(columns_per_row, max_images_in_cluster)  # Limit columns per row
    
    # Create subplots with enough space between them
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust space between plots
    
    # Ensure axes is a 2D array if there is only one row or one column
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # Convert to 2D if only 1 row
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)  # Convert to 2D if only 1 column
    
    # Iterate through each cluster
    for i, label in enumerate(unique_labels):
        cluster_indices = np.where(labels == label)[0]
        
        # Iterate through each image in the cluster
        for j, idx in enumerate(cluster_indices):
            # Ensure j doesn't exceed the number of columns
            if j < cols:
                ax = axes[i, j]  # Access the correct subplot
                ax.imshow(images[idx])
                ax.set_title(f"{filenames[idx]}", fontsize=8)  # Set a smaller font size for clarity
                ax.axis('off')  # Hide axes

        # Turn off unused axes if the cluster has fewer images than the columns
        for j in range(len(cluster_indices), cols):
            axes[i, j].axis('off')
    
    # Handle case when there are fewer clusters than rows
    if rows > num_clusters:
        for i in range(num_clusters, rows):
            for j in range(cols):
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

 
# Evaluate clustering performance with Precision, Recall, F1-Score

 
def evaluate_clustering(true_labels, predicted_labels):
    unique_labels = np.unique(predicted_labels)
    label_mapping = {}

    for label in unique_labels:
        # Find the true labels that are closest to the predicted label
        mask = (predicted_labels == label)
        closest_true_labels = true_labels[mask]
        most_common_true_label = np.bincount(closest_true_labels).argmax()
        label_mapping[label] = most_common_true_label

    # Map predicted labels to true labels
    adjusted_predicted_labels = np.array([label_mapping[label] for label in predicted_labels])

    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)
    accuracy = accuracy_score(true_labels, adjusted_predicted_labels)  # Accuracy computation

    return precision, recall, f1, accuracy

 
def plot_evaluation_metrics(precision, recall, f1, accuracy):
    # Metrics names
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values = [precision, recall, f1, accuracy]

    # Create the bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylim(0, 1)  # Set the y-axis limits to be between 0 and 1
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Clustering Evaluation Metrics')

    # Display the score on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

 
# Main function

 
def main():
    # Set the folder path where your images are stored
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(cur_dir, 'CatsDogsDataset', 'train')
    test_folder_path  = os.path.join(cur_dir, 'CatsDogsDataset', 'test')
    
    # Load images from folder
    images, filenames = load_images_from_folder(folder_path)

    # Now load testing data
    test_images, test_true_labels, test_filenames = load_images_and_labels_from_folder(test_folder_path)

    # Preprocess images by flattening them
    features = preprocess_images(images)

    # Preprocess test images (flatten)
    test_features = preprocess_images(test_images)    

    # Perform KMeans clustering
    start_time = time.time()
    num_clusters = 2  # Specify the number of clusters
    kmeans = perform_kmeans_clustering(features, num_clusters)
    end_time = time.time()
    train_runtime = end_time - start_time
    
    # Display clustered images
    display_clusters(images, kmeans, filenames)

    # Predict clusters for test data using the trained KMeans model
    test_predicted_labels = kmeans.predict(test_features)

    # Evaluate the clustering performance on the test data
    precision, recall, f1, accuracy = evaluate_clustering(test_true_labels, test_predicted_labels)

    # Print results
    print(f"Training Runtime: {train_runtime:.4f} seconds")
    print(f"Precision (Test): {precision:.4f}")
    print(f"Recall (Test): {recall:.4f}")
    print(f"F1-Score (Test): {f1:.4f}")
    print(f"Accuracy (Test): {accuracy:.4f}")

    plot_evaluation_metrics(precision, recall, f1, accuracy)

 
if __name__ == "__main__":
    main()


