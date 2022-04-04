
import cv2
import mediapipe as mp




class HandDetector:

    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

        #hardcoded id of the tips
        self.tipIds = [4,8,12,16,20]

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) 


        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
                print(handLms)
        return img

    def findPosition(self,img,handNo=0,draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for idx, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([idx,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)

        return lmList


    def fingersUp(self,lmList):
        fingers = []
        y_value = 2

        #thump
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0]-1][1]:
            fingers.append(0)
        else:
            fingers.append(1)


        for fingersTip in self.tipIds[1:]:
            if lmList[fingersTip][y_value] < lmList[fingersTip-1][y_value]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
            

