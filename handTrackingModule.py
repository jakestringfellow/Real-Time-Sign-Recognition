'''
Jake Stringfellow
Real-Time Sign Language Recognition
Building, training, analyzing, and modifying a deep network for real-time hand gesture
recognition based on the ASL (American Sign Language) alphabet.

handTrackingModule.py: Hand-Tracking Module based on the hand landmarks detection guide provided by MediaPipe
'''

# import statements
import cv2
import mediapipe as mp

'''
    handTracker
    Hand Tracking module based on the hand landmarks detection guide provided by MediaPipe.
    Used to localize key points of hands by using a ML model to operate on image data and output 
    landmarks as image coordinates.
    
    Constructor for handtracker, provides the basic parameters required for hand tracking
'''
class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode                            # If false, treats input images as video stream
        self.maxHands = maxHands                    # Max number of hands to detect
        self.detectionCon = detectionCon            # Minimum detection confidence
        self.modelComplex = modelComplexity         # Complexity of the hand or landmark
        self.trackCon = trackCon                    # Minimum tracking confidence
        self.mpHands = mp.solutions.hands           # MediaPipe hand tracking solution
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils    # Shortcut to mp drawing functions

    '''
    handsFinder
    Method that tracks the hands given an input image
    Params: 
        image: input image where hand tracking is done
        draw: Default to true, draws hand "skeleton" over found hand
    Returns: 
        image: output image with landmarks drawn
    '''
    def handsFinder(self, image, draw=True):
        # Convert BGR image to RGB
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imageRGB)

        # Identify and draw each hand landmark in the image, as well as the connections between them
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

        return image

    '''
    positionFinder
    Method to find x and y coordinates of each hand point (landmark)
    Params:
        image: input image where hands are being tracked
        handNo: Finds position for (default first) hand found in image
        draw: boolean for drawing, default true
    Returns:
        lmlist, list of lists of landmark id and coordinates
    '''
    def positionFinder(self, image, handNo=0, draw=True):
        # List of landmarks
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])


        return lmlist
