import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Directory containing the images
dataset_dir = 'dataset'  # Change this to your images directory if needed

# Function to load images and their labels
def load_data(dataset_dir):
    images = []
    labels = []
    # Loop through all the image files
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.png'):
            # Extract label from the filename, assuming labels are in the format 'label_X'
            parts = filename.split('_')
            label = int(parts[-1].replace('.png', ''))  # Extract label from filename
            
            # Read image
            img = cv2.imread(os.path.join(dataset_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))  # Resize the images to 28x28
            images.append(img)
            labels.append(label)
    
    # Convert to numpy arrays and normalize pixel values
    images = np.array(images) / 255.0  # Normalize to [0, 1]
    labels = np.array(labels)
    
    return images, labels

# Load dataset
X, y = load_data(dataset_dir)

# Reshape X for the CNN input
X = X.reshape(X.shape[0], 28, 28, 1)  # Adding the channel dimension (1 for grayscale)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

train_datagen.fit(X_train)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 output neurons for digits 0-9
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
          epochs=2000,
          validation_data=(X_test, y_test))

# Save the trained model
model.save('voltage_digit_recognition_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')
