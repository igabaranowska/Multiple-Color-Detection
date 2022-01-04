"""
@author: igabaranowska
"""
# Python code for Multiple Color Detection

from FoxDot import *
import numpy as np
import cv2

# For sound synthesis
i = 0

# Capturing video through webcam
liveCamera = cv2.VideoCapture(0)

# Start a while loop
while (1):

    # Reading the video from the
    # webcam in image frames
    _, imageFrame = liveCamera.read()

    # initialize lists to store characteritics of detected objects
    W = []; X = []; Y = []; H = [];

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    # cv2.cvtColor() function converts colorspace
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # the lower and upper are the color's boundaries
    # inRange() function returns a binary mask of the frame where the color is present

    # Set range for red color and
    # define mask
    redLower = np.array([136, 87, 111], np.uint8)
    redUpper = np.array([180, 255, 255], np.uint8)
    redMask = cv2.inRange(hsvFrame, redLower, redUpper)

    # Set range for green color and
    # define mask
    greenLower = np.array([25, 52, 72], np.uint8)
    greenUpper = np.array([102, 255, 255], np.uint8)
    greenMask = cv2.inRange(hsvFrame, greenLower, greenUpper)

    # Set range for blue color and
    # define mask
    blueLower = np.array([94, 80, 2], np.uint8)
    blueUpper = np.array([120, 255, 255], np.uint8)
    blueMask = cv2.inRange(hsvFrame, blueLower, blueUpper)

    # Set range for green color and
    # define mask
    greenLowertest = np.array([12, 26, 36], np.uint8)
    greenUppertest = np.array([51, 127, 127], np.uint8)
    greenMasktest = cv2.inRange(hsvFrame, greenLowertest, greenUppertest)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    # "uint8" means 8 bit integer matrix
    kernel = np.ones((5, 5), np.uint8)

    # For red color
    redMask = cv2.dilate(redMask, kernel)
    resRed = cv2.bitwise_and(imageFrame, imageFrame, mask=redMask)

    # For green color
    greenMask = cv2.dilate(greenMask, kernel)
    resGreen = cv2.bitwise_and(imageFrame, imageFrame, mask=greenMask)

    greenMasktest = cv2.dilate(greenMasktest, kernel)
    resGreentest = cv2.bitwise_and(imageFrame, imageFrame, mask=greenMasktest)

    # For blue color
    blueMask = cv2.dilate(blueMask, kernel)
    resBlue = cv2.bitwise_and(imageFrame, imageFrame, mask=blueMask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(redMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Coordinates and characteristics of the object are stored in lists
            W.append(w);
            X.append(x + (w / 2));
            Y.append(y + (h / 2));
            H.append(h);

            cv2.putText(imageFrame, "Red Color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(greenMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Coordinates and characteristics of the object are stored in lists
            W.append(w);
            X.append(x + (w / 2));
            Y.append(y + (h / 2));
            H.append(h);

            # Display position
            cv2.putText(imageFrame, "Green Color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))


    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blueMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Coordinates and characteristics of the object are stored in lists
            W.append(w);
            X.append(x + (w / 2));
            Y.append(y + (h / 2));
            H.append(h);

            cv2.putText(imageFrame, "Blue Color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))



    # Program Termination
    cv2.imshow("Color Detection on live camera", imageFrame)


    # Sound synthesis
    if len(X) > 0:
        i = 0
        c1 >> play("o{ -~P}X{-~P}", amp=1, dur=1 / 2, chop=3)
        if len(X) > 1:
            c2 >> play("<o   >< g g><{SSM}>", amp=1).often("amen")
        else:
            c2.stop()

    else:
        if i > 20:
            c1.stop()
        else:
            i = i + 1


    # waitKey() means the window will be opened until 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        # Stop the FoxDot players
        Clock.clear()
        # Releasing all the resources
        liveCamera.release()
        cv2.destroyAllWindows()
        break
