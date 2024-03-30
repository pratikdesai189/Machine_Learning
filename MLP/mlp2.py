import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from joblib import load
from sklearn.metrics import ConfusionMatrixDisplay

# Step 1: Read Data from Text Files
occupied_file_path = r"/content/drive/MyDrive/Machine_Learning/Combined_empty3n.txt"
empty_file_path = r"/content/drive/MyDrive/Machine_Learning/Combined_empty3.txt"
# Load data function
def load_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        data = []
        rows = 0
        for line in lines:
            if line.strip():  # Check if the line is not empty
                rows += 1
                values = line.strip().split(",")  # Assuming data is comma-separated
                try:
                    # Convert strings to floats (if necessary)
                    data.append([float(value) for value in values])
                except ValueError:
                    print(f"Ignoring line {rows}: Unable to convert to floats:", values)
        return data

# Load data from files
empty_data = load_data(empty_file_path)
occupied_data = load_data(occupied_file_path)

# Combine the data from both files
combined_data = np.concatenate((empty_data, occupied_data), axis=0)

# Step 4: Load Pre-Trained Model
model_file = "/content/pratik_mlp_classifier_model.joblib"  # Path to the saved pre-trained model
loaded_model = load(model_file)  # Load the model using joblib

# Step 5: Make Predictions
new_predictions = loaded_model.predict(combined_data)
print("Predictions:", new_predictions)

# Create true labels based on the number of empty and occupied samples
true_labels = ['empty'] * len(empty_data)+  ['occupied'] * len(occupied_data)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, new_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate confusion matrix in percentage
total_samples = np.sum(conf_matrix)
cm_percent = (conf_matrix / total_samples) * 100
print("Confusion Matrix (Percentage):")
print(cm_percent)

# Visualize confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=['empty', 'occupied'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
