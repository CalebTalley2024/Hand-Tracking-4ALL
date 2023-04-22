# Import the necessary libraries.
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import tensorflow
import math

# Initialize the video capture object and the hand detector.
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load the classification model.
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# print('classifier works :)')
# Define some constants.
offset = 20  # The offset in pixels to crop the hand image.
# imgSize = 300  # The size of the cropped hand image.
imgSize = 224
folder = "alphabet_data/C"  # The folder to save the cropped hand images
folder = "other"
counter = 0  # The counter for the number of cropped hand images.

# Create a list of labels.
# labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "W",
#           "X", "Y", "Z"]

# 5 letter prototype
labels = ["A","B","C","D","E"]

# Start the main loop.
while True:
    # Read a frame from the video.
    success, img = cap.read()

    # If the frame was read successfully, then...
    if success:
        # Find the hands in the frame.
        hands, img = detector.findHands(img)

        # If hands were found, then...
        if hands:
            # Get the first hand.
            hand = hands[0]

            # Get the bounding box of the hand.
            x, y, w, h = hand['bbox']

            # Crop the hand image.
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            # Resize the cropped image to the desired size.
            imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
            # print(f"size of image reshaped {imgResize}")

            # Get the prediction from the classification model.
            prediction, index = classifier.getPrediction(imgResize, draw=False)

            # Display the prediction on the image.
            cv2.putText(img, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

            # Save the cropped image to a file.
            # cv2.imwrite(f"{folder}/{counter}.jpg", imgCrop)
            counter += 1

        # Display the image.
        cv2.imshow("Image", img)

    # Wait for a key press.
    key = cv2.waitKey(1)

    # If the `q` key was pressed, then...
    if key == ord("q"):
        # Break out of the main loop.
        break

# Release the video capture object.
cap.release()

# Close all open windows.
cv2.destroyAllWindows()