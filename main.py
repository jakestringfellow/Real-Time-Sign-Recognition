'''
Jake Stringfellow
Real-Time Sign Language Recognition
Building, training, analyzing, and modifying a deep network for real-time hand gesture
recognition based on the ASL (American Sign Language) alphabet.

main.py: The main file of the program. The main function loads the neural network trained for Sign language
         recognition, then through a live stream attempts to accurately predict the sign that the user is
         gesturing.
         Two windows are displayed, one being the live video feed with hand-tracking information and region of
         interest box drawn, and the other being the augmented live video being fed to the CNN.
         Displayed above the bounding box is the CNN's prediction of the presented sign.
'''

# Import statements
import cv2
import mediapipe as mp
import handTrackingModule
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import keras.utils as image
import imutils

# Global variables
CAPTURE_WIDTH = 900
ROI_LONG = 400  # Region Of Interest length across
ROI_MARGIN = 50
ROI_TOP = ROI_MARGIN
ROI_RIGHT = CAPTURE_WIDTH - ROI_MARGIN
ROI_BOTTOM = ROI_TOP + ROI_LONG
ROI_LEFT = ROI_RIGHT - ROI_LONG

def preprocessImage(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.bitwise_not(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=0.1)

    return gray

'''
    Main Function
    Capture's live video from user's webcam, initializes hand-tracking for recognized hands in the video stream,
    loads the Sign-MNIST trained CNN.
'''
def main():

    # Open the user video
    cap = cv2.VideoCapture(0)

    # Initialize our hand tracking module
    tracker = handTrackingModule.handTracker()

    # Load our trained model
    model = load_model('support/ASL_model', compile=False)

    # Obtain and print the available signs
    f = open('support/sign.names','r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)

    # Until the user quits
    while True:

        # Store the live feed in frames
        success,frame = cap.read()

        # Create the ROI where gesture recognition takes place
        cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (255,255,0), 2)
        roi = frame[ROI_TOP+2:ROI_BOTTOM-2, ROI_LEFT+2:ROI_RIGHT-2]

        # Preprocessing the image to more closely match the dataset it was trained on
        gray = preprocessImage(roi)

        # Display the processed image to the user in a separate window
        cv2.imshow("ROI", gray)

        # Resize the ROI image to resemble dataset images
        gray_small = imutils.resize(gray, 28)
        gray_small = gray_small.reshape(1,28,28,1)

        # Have the model make a prediction on what gesture the user is making
        prediction = model.predict([gray_small])
        # Label that prediction onto the top of the ROI
        predictionLabel = classNames[np.argmax(prediction)]

        # Use the handTrackingModule to find the hand in the frame and display landmarks
        # For future implementation this will be used to move the ROI with the user's hands
        frame = tracker.handsFinder(frame)
        lmList = tracker.positionFinder(frame)
        if len(lmList) != 0:
            print(lmList[9])

        labelColor = (0,255,0)
        cv2.putText(frame, predictionLabel.capitalize(), (ROI_LEFT, ROI_TOP - 7), cv2.FONT_HERSHEY_DUPLEX, 1, labelColor, 2)

        # Display the live video feed
        cv2.imshow("Video",frame)

        # Quit the program if the user presses "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()