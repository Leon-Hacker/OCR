This is a project that uses convolutional neural networks to recognize numbers displayed on an LED digital tube in videos. 
The video contains glare from the LEDs, which seriously interferes with OpenCV algorithms, but the convolutional neural network can effectively solve this problem.
Step 1: Extract ROI from video, preprocess and save to dataset;
Step 2：Train Convolutional Neural Networks；
Step 3: Use a trained model to identify LED digital tubes in the video.