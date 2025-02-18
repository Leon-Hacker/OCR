import os
import cv2
import pytesseract
import numpy as np
import tensorflow as tf

# Load the voltage digit recognition model
model = tf.keras.models.load_model('voltage_digit_recognition_model.h5')

# Define the ROI coordinates (time and voltage)
time_roi = (406, 4, 225, 47)  # (x, y, width, height) for time
voltage_roi_1st_digit = (581, 509, 80, 55)  # (x, y, width, height) for 1st digit of voltage
voltage_roi_2nd_digit = (573, 569, 79, 54)  # (x, y, width, height) for 2nd digit of voltage
voltage_roi_3rd_digit = (563, 626, 81, 54)  # (x, y, width, height) for 3rd digit of voltage

# Function to rotate an image by a specified angle
def rotate_image(image, angle):
    # Get image center
    center = (image.shape[1] // 2, image.shape[0] // 2)
    # Create a rotation matrix
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

# Function for preprocessing voltage ROI (rotation + color range thresholding)
def preprocess_voltage_roi(roi, angle=100):
    # Step 1: Rotate the image by the specified angle
    rotated_roi = rotate_image(roi, angle)
    
    # Step 2: Convert to HSV (Hue, Saturation, Value) color space for better handling of red areas
    hsv_image = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2HSV)
    
    # Step 3: Adjusted the lower and upper bounds to cover a wider range of red
    lower_red = np.array([0, 50, 50])  # Adjusted lower bound (H: 0, S: 50, V: 50)
    upper_red = np.array([15, 255, 255])  # Adjusted upper bound (H: 15, S: 255, V: 255)
    
    # Step 4: Create a mask for the red areas
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    
    # Step 5: Create an inverse mask to keep the rest of the image (digits)
    inverse_mask = cv2.bitwise_not(red_mask)
    
    # Step 6: Use the inverse mask to keep the white digits and preserve the red background
    white_digits = cv2.bitwise_and(rotated_roi, rotated_roi, mask=inverse_mask)
    
    # Step 7: Convert the red areas to black
    black_background = cv2.bitwise_and(rotated_roi, rotated_roi, mask=red_mask)
    
    # Step 8: Combine the processed regions: white digits (unchanged) and black background (for red areas)
    processed_image = cv2.add(white_digits, black_background)
    
    # Step 9: Convert to grayscale for neural network training
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # Step 10: Optional: Apply morphological operations to improve digit clarity
    kernel = np.ones((3, 3), np.uint8)
    final_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    
    # Step 11: Resize to match the input shape for the CNN model
    resized_image = cv2.resize(final_image, (28, 28))
    
    # Step 12: Normalize the pixel values
    normalized_image = resized_image / 255.0
    
    # Step 13: Reshape for the model (28x28x1)
    final_image = normalized_image.reshape(1, 28, 28, 1)

    return final_image

# Function to extract time from the time ROI using pytesseract
def extract_time_from_roi(frame):
    time_roi_image = frame[time_roi[1]:time_roi[1] + time_roi[3], time_roi[0]:time_roi[0] + time_roi[2]]
    # Convert to grayscale for OCR
    gray_image = cv2.cvtColor(time_roi_image, cv2.COLOR_BGR2GRAY)
    # Use pytesseract to extract text (time)
    extracted_time = pytesseract.image_to_string(gray_image, config='--psm 6')
    return extracted_time.strip()

# Function to process every 50th frame and recognize voltage and time data
def process_video_frames(video_path):
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Check if the video is opened
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return

    # Loop through every frame in the video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process every 50th frame
        if frame_count % 50 == 0:
            print(f"Processing frame {frame_count}...")

            # Extract time from the time ROI
            extracted_time = extract_time_from_roi(frame)
            print(f"Extracted Time: {extracted_time}")

            # Extract and preprocess voltage digits from the ROI
            voltage_roi_1st = frame[voltage_roi_1st_digit[1]:voltage_roi_1st_digit[1] + voltage_roi_1st_digit[3],
                                    voltage_roi_1st_digit[0]:voltage_roi_1st_digit[0] + voltage_roi_1st_digit[2]]
            voltage_roi_2nd = frame[voltage_roi_2nd_digit[1]:voltage_roi_2nd_digit[1] + voltage_roi_2nd_digit[3],
                                    voltage_roi_2nd_digit[0]:voltage_roi_2nd_digit[0] + voltage_roi_2nd_digit[2]]
            voltage_roi_3rd = frame[voltage_roi_3rd_digit[1]:voltage_roi_3rd_digit[1] + voltage_roi_3rd_digit[3],
                                    voltage_roi_3rd_digit[0]:voltage_roi_3rd_digit[0] + voltage_roi_3rd_digit[2]]

            # Preprocess each voltage ROI for prediction
            processed_1st = preprocess_voltage_roi(voltage_roi_1st)
            processed_2nd = preprocess_voltage_roi(voltage_roi_2nd)
            processed_3rd = preprocess_voltage_roi(voltage_roi_3rd)

            # Predict voltage digits using the model
            pred_1st = model.predict(processed_1st.reshape(1, 28, 28, 1))
            pred_2nd = model.predict(processed_2nd.reshape(1, 28, 28, 1))
            pred_3rd = model.predict(processed_3rd.reshape(1, 28, 28, 1))

            # Get the digit with the highest probability (from softmax)
            voltage_1st = np.argmax(pred_1st)
            voltage_2nd = np.argmax(pred_2nd)
            voltage_3rd = np.argmax(pred_3rd)

            print(f"Predicted Voltage: {voltage_1st}{voltage_2nd}{voltage_3rd}")

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    cap.release()

# Specify the video path
video_path = 'voltage_data.MP4'  # Change this to your video file path

# Call the function to process the video frames
process_video_frames(video_path)