# Import necessary libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to load FFT data from a folder and assign labels
def load_data(file_path, label):
    X = []
    y = []
    with open(file_path, "r") as f:
                # Iterate over each line in the file
                for line in f:
                    # Split the line by comma (or any other delimiter)
                    values = line.strip().split(",")
                    try:
                        # Convert string values to floats
                        data = [float(value) for value in values]
                        X.append(data)
                        y.append(label)
                    except ValueError:
                        print(f"Error converting values to float in file '{file_path}': {values}")
                        continue
    return X, y
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(folder_path, filename)
#             # Load data from the text file


# Directories containing FFT files for occupied and empty labels
occupied_path = r"/content/drive/MyDrive/Machine_Learning/Combined_empty3n.txt"
empty_path = r"/content/drive/MyDrive/Machine_Learning/Combined_empty3.txt"

# Load data for occupied and empty labels
X_occupied, y_occupied = load_data(occupied_path, label="occupied")
X_empty, y_empty = load_data(empty_path, label="empty")

# Combine data and labels
X = np.concatenate((X_occupied, X_empty), axis=0)
y = np.concatenate((y_occupied, y_empty), axis=0)

# Split the combined dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Evaluate the model
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Other metrics
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
total_samples = np.sum(cm)
cm_percent = (cm / total_samples) * 100

print("Confusion Matrix in Percentage):")
print(cm_percent)


from joblib import dump

# Save the trained model to a file
model_file = "pratik_mlp_classifier_model.joblib"
dump(mlp, model_file)