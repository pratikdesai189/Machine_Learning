import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

empty_file_path = r"/content/drive/MyDrive/Machine_Learning/Combined_empty3.txt"
occupied_file_path = r"/content/drive/MyDrive/Machine_Learning/Combined_empty3n.txt"

def load_data(file_path, label):
    X = []
    y = []
    with open(file_path, "r") as f:
        for line in f:
            values = line.strip().split(",")
            try:
                data = [float(value) for value in values]
                X.append(data)
                if label == "occupied":
                    y.append(1)  # Assign label 1 for 'occupied'
                elif label == "empty":
                    y.append(0)  # Assign label 0 for 'empty'
                else:
                    raise ValueError("Invalid label specified.")
            except ValueError:
                print(f"Error converting values to float in file '{file_path}': {values}")
                continue
    return X, y

X_occupied, y_occupied = load_data(occupied_file_path, label="occupied")  # Label 1 for 'occupied'
X_empty, y_empty = load_data(empty_file_path, label="empty")  # Label 0 for 'empty'

X_occupied, y_occupied = load_data(occupied_file_path, label="occupied")  # Label 1 for 'occupied'
X_empty, y_empty = load_data(empty_file_path, label="empty")  # Label 0 for 'empty'

X = np.concatenate((X_occupied, X_empty), axis=0)
y = np.concatenate((y_occupied, y_empty), axis=0)

# Normalize the data
X = X / np.max(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary classification loss
              metrics=['accuracy'])

# Train the model
model.fit(np.expand_dims(X_train, axis=-1), y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(np.expand_dims(X_test, axis=-1), y_test)
print("Test accuracy:", test_acc)