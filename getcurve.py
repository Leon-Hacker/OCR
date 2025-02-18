import os
import cv2
import pytesseract
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the voltage digit recognition model
model = tf.keras.models.load_model('voltage_digit_recognition_model.h5')

# Define the ROI coordinates (time and voltage)
time_roi = (406, 4, 225, 47)  # (x, y, width, height) for time
voltage_roi_1st_digit = (581, 509, 80, 55)  # (x, y, width, height) for 1st digit of voltage
voltage_roi_2nd_digit = (573, 569, 79, 54)  # (x, y, width, height) for 2nd digit of voltage
voltage_roi_3rd_digit = (563, 626, 81, 54)  # (x, y, width, height) for 3rd digit of voltage

# Function to rotate an image by a specified angle
def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

# Function for preprocessing voltage ROI (rotation + color range thresholding)
def preprocess_voltage_roi(roi, angle=100):
    rotated_roi = rotate_image(roi, angle)
    hsv_image = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])  
    upper_red = np.array([15, 255, 255]) 
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    inverse_mask = cv2.bitwise_not(red_mask)
    white_digits = cv2.bitwise_and(rotated_roi, rotated_roi, mask=inverse_mask)
    black_background = cv2.bitwise_and(rotated_roi, rotated_roi, mask=red_mask)
    processed_image = cv2.add(white_digits, black_background)
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    final_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    resized_image = cv2.resize(final_image, (28, 28))
    normalized_image = resized_image / 255.0
    final_image = normalized_image.reshape(1, 28, 28, 1)
    return final_image

# Function to extract time from the time ROI using pytesseract
def extract_time_from_roi(frame):
    time_roi_image = frame[time_roi[1]:time_roi[1] + time_roi[3], time_roi[0]:time_roi[0] + time_roi[2]]
    gray_image = cv2.cvtColor(time_roi_image, cv2.COLOR_BGR2GRAY)
    extracted_time = pytesseract.image_to_string(gray_image, config='--psm 6')
    return extracted_time.strip()

# Function to process every 50th frame and recognize voltage and time data
def process_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return

    frame_count = 0
    times = []
    voltages = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

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
            pred_1st = model.predict(processed_1st)
            pred_2nd = model.predict(processed_2nd)
            pred_3rd = model.predict(processed_3rd)

            voltage_1st = np.argmax(pred_1st)
            voltage_2nd = np.argmax(pred_2nd)
            voltage_3rd = np.argmax(pred_3rd)

            voltage = int(f"{voltage_1st}{voltage_2nd}{voltage_3rd}")
            print(f"Predicted Voltage: {voltage}")
            
            # Store extracted time and voltage for plotting
            times.append(extracted_time)
            voltages.append(voltage)

        frame_count += 1

    # Release the video capture object
    cap.release()

    # Plotting the relationship between voltage and time
    plot_voltage_vs_time(times, voltages)

# Function to plot voltage vs time
def plot_voltage_vs_time(times, voltages):
    # Convert times to a format that can be used for plotting (e.g., hours, minutes, seconds)
    time_hours = []
    for time in times:
        if len(time) >= 8:  # If the time string is in HH:MM:SS format
            h, m, s = time.split(':')
            total_seconds = int(h) * 3600 + int(m) * 60 + int(s)
            time_hours.append(total_seconds)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(time_hours, voltages, marker='o', color='b', label='Voltage')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage')
    plt.title('Voltage vs Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Specify the video path
video_path = 'voltage_data.MP4'  # Change this to your video file path

# Call the function to process the video frames
process_video_frames(video_path)
