import os
import cv2
import numpy as np
import pandas as pd

# Define the ROI coordinates (from your provided values)
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

# Function for preprocessing image (color range thresholding)
def preprocess_image(image):
    # Convert to HSV (Hue, Saturation, Value) color space for better handling of red areas
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Adjusted the lower and upper bounds to cover a wider range of red
    lower_red = np.array([0, 50, 50])  # Adjusted lower bound (H: 0, S: 50, V: 50)
    upper_red = np.array([15, 255, 255])  # Adjusted upper bound (H: 15, S: 255, V: 255)
    
    # Create a mask for the red areas
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    
    # Create an inverse mask to keep the rest of the image (digits)
    inverse_mask = cv2.bitwise_not(red_mask)
    
    # Use the inverse mask to keep the white digits and preserve the red background
    white_digits = cv2.bitwise_and(image, image, mask=inverse_mask)
    
    # Convert the red areas to black
    black_background = cv2.bitwise_and(image, image, mask=red_mask)
    
    # Combine the processed regions: white digits (unchanged) and black background (for red areas)
    processed_image = cv2.add(white_digits, black_background)
    
    # Convert to grayscale for neural network training
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # Optional: apply morphological operations to improve digit clarity
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    
    return processed_image

# Function to collect dataset from the 100th frame
def collect_dataset(video_path, output_folder, target_frame=100):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open video.")
        exit()

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set the video capture to the 100th frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)

    # Read the 100th frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Couldn't read the {target_frame}th frame.")
        exit()

    # Extract Voltage ROIs (before rotation)
    voltage_roi_1st = frame[voltage_roi_1st_digit[1]:voltage_roi_1st_digit[1] + voltage_roi_1st_digit[3],
                            voltage_roi_1st_digit[0]:voltage_roi_1st_digit[0] + voltage_roi_1st_digit[2]]
    voltage_roi_2nd = frame[voltage_roi_2nd_digit[1]:voltage_roi_2nd_digit[1] + voltage_roi_2nd_digit[3],
                            voltage_roi_2nd_digit[0]:voltage_roi_2nd_digit[0] + voltage_roi_2nd_digit[2]]
    voltage_roi_3rd = frame[voltage_roi_3rd_digit[1]:voltage_roi_3rd_digit[1] + voltage_roi_3rd_digit[3],
                            voltage_roi_3rd_digit[0]:voltage_roi_3rd_digit[0] + voltage_roi_3rd_digit[2]]

    # Rotate the voltage regions by 100 degrees
    voltage_roi_1st_rotated = rotate_image(voltage_roi_1st, 100)
    voltage_roi_2nd_rotated = rotate_image(voltage_roi_2nd, 100)
    voltage_roi_3rd_rotated = rotate_image(voltage_roi_3rd, 100)

    # Preprocess the rotated voltage regions
    voltage_roi_1st_processed = preprocess_image(voltage_roi_1st_rotated)
    voltage_roi_2nd_processed = preprocess_image(voltage_roi_2nd_rotated)
    voltage_roi_3rd_processed = preprocess_image(voltage_roi_3rd_rotated)

    # Display the preprocessed voltage images
    cv2.imshow("1st Digit of Voltage", voltage_roi_1st_processed)
    cv2.imshow("2nd Digit of Voltage", voltage_roi_2nd_processed)
    cv2.imshow("3rd Digit of Voltage", voltage_roi_3rd_processed)

    # Wait for user to press keys to input the labels
    key = cv2.waitKey(0)
    if key == 27:  # Escape key to exit
        return

    # Manually label the digits
    print("Please input the labels for the 3 digits (0-9) separated by space:")
    label_1st = int(input("Label for the 1st digit: "))
    label_2nd = int(input("Label for the 2nd digit: "))
    label_3rd = int(input("Label for the 3rd digit: "))

    # Save the processed images and labels to the output folder
    dataset = []
    dataset.append((voltage_roi_1st_processed, label_1st))
    dataset.append((voltage_roi_2nd_processed, label_2nd))
    dataset.append((voltage_roi_3rd_processed, label_3rd))

    # Save images and labels
    for i, (image, label) in enumerate(dataset):
        filename = os.path.join(output_folder, f"frame_{target_frame}_voltage_digit_{i+1}_label_{label}.png")
        cv2.imwrite(filename, image)  # Save images
        print(f"Saved {filename}")

    # Optionally, save labels to a CSV
    labels_df = pd.DataFrame(dataset, columns=["Image", "Label"])
    labels_df.to_csv(os.path.join(output_folder, "voltage_labels.csv"), mode='a', header=False, index=False)

    # Close the OpenCV windows
    cv2.destroyAllWindows()

    # Release the video capture object
    cap.release()

# Specify the video path and output folder
video_path = 'voltage_data.MP4'  # Change this to your video file
output_folder = 'dataset'  # Change this to your desired output folder

# Call the function to collect the dataset from the 100th frame
collect_dataset(video_path, output_folder, target_frame=450)
