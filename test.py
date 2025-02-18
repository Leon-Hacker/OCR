import os
import cv2
import pytesseract
import numpy as np

# Set the TESSDATA_PREFIX to the correct Tesseract directory (not tessdata folder)
os.environ['TESSDATA_PREFIX'] = r'C:/Program Files/Tesseract-OCR/tessdata'  

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
    
    # Convert to grayscale for OCR
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    
    # Optional: apply morphological operations to improve digit clarity
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    
    return processed_image

# Load the video
video_path = 'voltage_data.MP4'  # Change this to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

# Read the first frame to allow for ROI selection
ret, frame = cap.read()
if not ret:
    print("Error: Couldn't read the video frame.")
    exit()

# Extract Time ROI
x, y, w, h = time_roi
time_roi_frame = frame[y:y+h, x:x+w]

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

# Apply OCR to the time region
time_text = pytesseract.image_to_string(time_roi_frame, config='--psm 6').strip()

# Apply OCR to the processed voltage regions
voltage_1st_digit = pytesseract.image_to_string(voltage_roi_1st_processed, config='--psm 7 --oem 3').strip()
voltage_2nd_digit = pytesseract.image_to_string(voltage_roi_2nd_processed, config='--psm 7 --oem 3').strip()
voltage_3rd_digit = pytesseract.image_to_string(voltage_roi_3rd_processed, config='--psm 7 --oem 3').strip()

# Combine the voltage digits
voltage = voltage_1st_digit + voltage_2nd_digit + voltage_3rd_digit

# Print the extracted time and voltage data
print(f"Extracted Time: {time_text}")
print(f"Extracted Voltage: {voltage}")

# Optional: Display the results for debugging (you can remove this)
cv2.imshow("Time ROI", time_roi_frame)
cv2.imshow("Processed 1st Digit of Voltage", voltage_roi_1st_processed)
cv2.imshow("Processed 2nd Digit of Voltage", voltage_roi_2nd_processed)
cv2.imshow("Processed 3rd Digit of Voltage", voltage_roi_3rd_processed)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video capture object
cap.release()
