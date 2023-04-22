import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time



# takes a picture and places it into folder
def take_picture(key, counter, folder):
    
    
    folders = np.array(["a","b","c","d","e"])

    # ascii values for the strings of numbers 1 - 5
    valid_keys = [ord(str(i)) for i in range(1, 6)]
    # if you click a key in folders, you will change the selected folder
    if key in valid_keys:

        # change the ord acsii string to it's keyboard key integer
        index_int= int(chr(key))
        folder_name = folders[index_int-1]
        folder = f"alphabet_data/{folder_name}"
        print(f"new folder selected:{folder_name}")
    # if you click "s", you will save a picture in the selected folder
    elif key == ord("s"):  # if the "s" key is pressed, save the resized hand image to a file
        counter += 1  # increment the counter
        print(f"pictures taken: ",counter)
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # save the image to a file with 
    elif key != -1: # -1 means that nothing is being pressed
        # print("click something else")
        print( chr(key))

    
    return key, counter, folder


cap = cv2.VideoCapture(0)  # initialize video capture object to read from the default camera
detector = HandDetector(maxHands=1)  # initialize a HandDetector object to detect hand landmarks in the captured frames

offset = 20  # set the offset value for cropping the hand region
# imgSize = 300  # set the size of the output image after resizing
imgSize = 224

folder = "other"  # set the folder to store the captured image
counter = 0  # set the counter for the number of images captured

# cv2 = 4.6.0
#

imgWhite = None
while True:
    success, img = cap.read()  # read a frame from the camera
    if success:  # check if the frame was read successfully
        hands, img = detector.findHands(img)  # use the HandDetector object to detect the hand landmarks in the frame
        if hands:  # check if a hand was detected
            hand = hands[0]  # get the first detected hand

            # x,y: The x and y-coordinate of the top-left corner of the bounding box relative to the top-left corner of the screen
            # w: The width of the bounding box.
            # h: The height of the bounding box.
            x, y, w, h = hand['bbox']  # get the bounding box of the hand

            if(x<20):
                print("x-dim:", x)

            if(y<20):
                print("y-dim:", y)

            # print("width:", w)
            # print("height:", h )

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # create a white image of size imgSize x imgSize x 3

            # if(x < 20 or y <20): # apply an offset if the x and y displacement is large enough
            #      imgCrop = img[y:y + h, x:x + w]  # crop the hand region from the frame
            # else:   
            #      imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # crop the hand region from the frame

            # make sure your hand is in img, if it leaves on the left or top side, the program will crash
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            # print(img.shape)
            # print(imgCrop.shape)

            aspectRatio = h / w  # calculate the aspect ratio of the hand region

            if aspectRatio > 1:  # if the aspect ratio is greater than 1, resize the hand image horizontally
                k = imgSize / h  # calculate the scaling factor
                wCal = math.ceil(k * w)  # calculate the new width
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # resize the image to the new dimensions
                wGap = math.ceil((imgSize - wCal) / 2)  # calculate the gap to center the resized image horizontally
                imgWhite[:, wGap:wCal + wGap] = imgResize  # copy the resized image to the center of the white image

            else:  # if the aspect ratio is less than or equal to 1, resize the hand image vertically
                k = imgSize / w  # calculate the scaling factor
                hCal = math.ceil(k * h)  # calculate the new height
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # resize the image to the new dimensions
                hGap = math.ceil((imgSize - hCal) / 2)  # calculate the gap to center the resized image vertically
                imgWhite[hGap:hCal + hGap, :] = imgResize  # copy the resized image to the center of the white image

            cv2.imshow("ImageCrop", imgCrop)  # display the cropped hand image
            cv2.imshow("ImageWhite", imgWhite)  # display the resized hand image on a white background

    cv2.imshow("Image", img)  # display the original frame with the detected hand
    key = cv2.waitKey(1)  # wait for a key press
    # print(key)
    key, counter, folder = take_picture(key, counter,folder)
        
# click 1,2,3,4, or 5 to get to the a,b,c,d, or e folders respectively


