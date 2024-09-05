import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.models import Model
from keras.utils import to_categorical

def ela_image(image_path, quality=90):
    """Perform Error Level Analysis on an image."""
    original = Image.open(image_path)
    
    # Save the image at the specified quality level
    resaved_path = "resaved.jpg"
    original.save(resaved_path, 'JPEG', quality=quality)

    # Re-open the saved image
    resaved = Image.open(resaved_path)

    # Find the difference between the original and the resaved image
    diff = ImageChops.difference(original, resaved)

    # Enhance the difference image to amplify the error levels
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    return np.array(diff)

def load_frames_with_ela(subfolder_path):
    """Load all frames and perform ELA from the 'frames' directory within a subfolder."""
    ela_frames = []
    frames_dir = os.path.join(subfolder_path, "frames")
    
    for filename in sorted(os.listdir(frames_dir)):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust for your file types
            frame_path = os.path.join(frames_dir, filename)
            ela_frame = ela_image(frame_path)
            ela_frame = cv2.resize(ela_frame, (224, 224))  # Resize to ResNet input size
            ela_frames.append(ela_frame)
    
    return ela_frames

def load_labels_from_file(file_path):
    """Load the video-level label (real or fake) from the label file."""
    with open(file_path, 'r') as file:
        label = file.readline().strip().split()[1]  # Assuming the label is on the first line
    return 1 if label == "fake" else 0  # Convert to binary label

def load_dataset_from_main_folder(main_folder_path):
    """Load video frames (with ELA) and labels from the main folder."""
    all_videos = []
    all_labels = []
    
    for subfolder_name in sorted(os.listdir(main_folder_path)):
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            ela_frames = load_frames_with_ela(subfolder_path)
            label = load_labels_from_file(os.path.join(subfolder_path, "labels.txt"))
            
            all_videos.append(np.array(ela_frames))
            all_labels.append(label)
    
    return np.array(all_videos), np.array(all_labels)

# Load dataset
main_folder_path = 'C:\\DEEPFAKE DETECTION\\dataset'
videos, labels = load_dataset_from_main_folder(main_folder_path)

# Normalize pixel values
normalized_videos = np.array([preprocess_input(video) for video in videos])

# Load pre-trained ResNet-18 model (ResNet50 with only 18 layers will be used as an example here)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Create a model to extract features before the fully connected layers
feature_extractor = Model(inputs=resnet_model.input, outputs=resnet_model.layers[-2].output)

# Extract features for each frame in each video
extracted_features = []
for video in normalized_videos:
    features = []
    for frame in video:
        frame = np.expand_dims(frame, axis=0)
        feature = feature_extractor.predict(frame)
        features.append(feature)
    extracted_features.append(np.squeeze(np.array(features)))

# Reshape to (num_samples, num_frames, feature_dim)
X = np.array(extracted_features)
y = to_categorical(labels, num_classes=2)  # Convert to one-hot encoding with two classes

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create the model
model = Sequential()
model.add(TimeDistributed(Dense(units=256, activation='relu'), input_shape=(X.shape[1], X.shape[2], X.shape[3])))
model.add(LSTM(units=128, return_sequences=True))
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
