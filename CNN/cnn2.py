import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf

# Data cleaning function
def clean_data(file_path, clean_file_path):
    with open(file_path, "r") as f, open(clean_file_path, "w") as out_f:
        for line in f:
            if '\x1a' not in line:
                out_f.write(line)
            else:
                cleaned_line = line.replace('\x1a', '')  # Replace with an empty string or other placeholder
                out_f.write(cleaned_line)

# Enhanced load data function
def load_data(file_path):
    data = []
    rows = 0
    with open(file_path, "r") as file:
        for line in file:
            rows += 1
            if line.strip():  # Check if the line is not empty
                values = line.strip().split(",")  # Assuming data is comma-separated
                try:
                    # Convert strings to floats (if necessary)
                    data.append([float(value) for value in values])
                except ValueError:
                    print(f"Ignoring line {rows}: Unable to convert to floats:", values)
    return data

# Define file paths
empty_file_path = r"/content/drive/MyDrive/Machine_Learning/Combined_empty3.txt"
occupied_file_path = r"/content/drive/MyDrive/Machine_Learning/Combined_empty3n.txt"

# Clean data files
clean_empty_file_path = empty_file_path.replace('.txt', '_cleaned.txt')
clean_occupied_file_path = occupied_file_path.replace('.txt', '_cleaned.txt')
clean_data(empty_file_path, clean_empty_file_path)
clean_data(occupied_file_path, clean_occupied_file_path)

# Load data from cleaned files
occupied_data = load_data(clean_occupied_file_path)
empty_data = load_data(clean_empty_file_path)

# Combine the data from both files
combined_data = np.concatenate((occupied_data, empty_data), axis=0)

# Load pre-trained model
model_file = "/content/pratik_model_new.keras"
loaded_model = tf.keras.models.load_model(model_file)

# Make predictions
new_predictions = loaded_model.predict(combined_data)
print("Predictions:", new_predictions)

# Create and process labels
true_labels = ['occupied'] * len(occupied_data) + ['empty'] * len(empty_data)
predicted_labels = ['empty' if pred < 0.5 else 'occupied' for pred in np.squeeze(new_predictions)]

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=['empty', 'occupied'])
print("Confusion Matrix:")
print(conf_matrix)

# Display confusion matrix as percentages
cm_percent = (conf_matrix / np.sum(conf_matrix)) * 100
print("Confusion Matrix (Percentage):")
print(cm_percent)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['empty', 'occupied'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()