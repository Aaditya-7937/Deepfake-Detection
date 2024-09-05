import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Define your function to load frames and labels
def load_frames_from_directory(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust for your file types
            frame = cv2.imread(os.path.join(directory, filename))
            frame = cv2.resize(frame, (64, 64))  # Resize to your desired dimensions
            frames.append(frame)
    return frames

def load_labels_from_file(file_path):
    with open(file_path, 'r') as file:
        labels = file.read().splitlines()
    return [label.split()[1] for label in labels]  # Extract the actual label ("fake")

# Load frames and labels from your dataset
frames_dir = 'C:\\DEEPFAKE DETECTION\\frames_test'
frames = load_frames_from_directory(frames_dir)
labels = load_labels_from_file("C:\\DEEPFAKE DETECTION\\labels.txt")
######################################
# Load real frames and labels
real_frames_dir = 'C:\\DEEPFAKE DETECTION\\real_frames'
real_frames = load_frames_from_directory(real_frames_dir)
real_labels = load_labels_from_file("C:\\DEEPFAKE DETECTION\\real_labels.txt")

# Normalize real pixel values
normalized_real_frames = np.array([frame / 255.0 for frame in real_frames])

# Reshape real input data
real_X = np.array([normalized_real_frames[i:i+num_frames].reshape(-1, feature_dim) for i in range(len(normalized_real_frames) - num_frames + 1)])
real_y = real_labels[num_frames-1:num_frames-1+len(real_X)]  # Adjust labels accordingly

# Manually encode real labels into a two-class one-hot format
real_y = [0 if label == "real" else 1 for label in real_y]  # Convert "real" to 0, "fake" to 1
real_y = to_categorical(real_y, num_classes=2)  # Convert to one-hot encoding with two classes

# Append real data to the existing dataset
X_temp = np.concatenate((X_temp, real_X))
y_temp = np.concatenate((y_temp, real_y))

# Split the combined dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Train the model with the combined dataset
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)
##################################################################################
# Normalize pixel values
normalized_frames = np.array([frame / 255.0 for frame in frames])

# Define num_frames and feature_dim
num_frames = 30  # Example number of frames
feature_dim = 64 * 64 * 3  # Example for a 64x64 RGB frame

# Reshape input data to the format (num_samples, num_frames, feature_dim)
X = np.array([normalized_frames[i:i+num_frames].reshape(-1, feature_dim) for i in range(len(normalized_frames) - num_frames + 1)])
y = labels[num_frames-1:num_frames-1+len(X)]  # Adjust labels accordingly

# Manually encode labels into a two-class one-hot format
y = [1 if label == "fake" else 0 for label in y]  # Convert "fake" to 1, "real" to 0
y = to_categorical(y, num_classes=2)  # Convert to one-hot encoding with two classes

# Verify the shape of y
print(f"Shape of y after one-hot encoding: {y.shape}")

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create the model
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(num_frames, feature_dim)))
model.add(LSTM(units=64))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax'))  # 2 units for the two classes: "fake" and "real"

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)
