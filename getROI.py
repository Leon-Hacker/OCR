import cv2

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

# Select ROI from the first frame
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)

# Print the ROI coordinates (x, y, width, height)
print("Selected ROI coordinates:", roi)

# Close the ROI selection window
cv2.destroyAllWindows()

# Now you can use these coordinates (roi) to extract the ROI from each frame of the video
# roi = (x, y, width, height)

# Extract the selected ROI from the first frame (for testing)
x, y, w, h = roi
roi_frame = frame[y:y+h, x:x+w]

# Show the selected ROI for verification
cv2.imshow("ROI", roi_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, you can save the ROI coordinates to use in the video processing
# If needed, save the ROI for further processing:
# You can later use these coordinates (x, y, w, h) in the video frame processing loop

# Release the video capture object
cap.release()
