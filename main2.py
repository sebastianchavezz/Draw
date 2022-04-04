import cv2
from HandTrackingModule import HandDetector
import time
import numpy as np
import math

global drawColor

def check_for_cicle_rock(fingersUp):
    if fingersUp[1] == 0 or fingersUp[4] == 0:
        return False
    for i,f in enumerate(fingersUp):
        if (i != 1 and i!=4) and f==1:
            return False
    return True  
def check_for_circle(fingersUp):
    for i,f in enumerate(fingersUp):
        if f == 0:
            return False
    return True

def determineDistance(lmList):
    x1= lmList[5][1]
    y1 = lmList[5][2]
    x2= lmList[18][1]
    y2 = lmList[18][2] 
    distance = math.sqrt((y2-y1)**2+(x2-x1)**2)
    return distance

def distanceBalls(distance):
    a = distance*2
    afstand = a /6
    return int(afstand)


def check_index_finger_up(fingersUp):
    index_finger = 1
    if fingersUp[index_finger] == 0:
        return False
    for i,f in enumerate(fingersUp):
        if i != 1 and f ==1:
            return False
    return True



def determineBrushThickness(lmList):
    x1= lmList[5][1]
    y1 = lmList[5][2]
    x2= lmList[17][1]
    y2 = lmList[17][2] 
    distance = math.sqrt((y2-y1)**2+(x2-x1)**2)
    
    brushThicness = distance*1/4

    #alpha = sigmoid(distance)
    return int(brushThicness)

def ballGetsBigger(distance):
    factor = distance//3
    return int(factor)

def check_the_color(lijst,diameter,middle_x):
    colors = [(255,255,255),(255,255,0),(255,0,255),(0,255,255)]
    radius = diameter//2
    for i,x_value in enumerate(lijst):
        if x_value-radius <= middle_x <= x_value+radius:
            return colors[i]
    return (0,0,0)

def pinky_is_up(fingersUp):
    pinky = 4
    if fingersUp[pinky] == 0:
        return False
    for i,f in enumerate(fingersUp):
        if i != pinky and f ==1:
            return False
    return True


def main():
        #fps
    pTime = 0
    cTime = 0
    #dimensions need to be the same
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    detector = HandDetector()
    
    
    img_previous = np.zeros((720,1280,3),np.uint8)
    drawColor = (0,0,0)
    brush_thickness = 5
    
    x_previous,y_previous = 0,0
    while True:
        # 1 Import Images
        succes, img = cap.read()
        img= cv2.flip(img,1)
        
        #2. Find Hand Landmark
        img = detector.findHands(img)
        lmlist= detector.findPosition(img)
        if len(lmlist) !=0:
            
            x_index ,y_index = lmlist[8][1:]
            x_pink , y_pink = lmlist[20][1:]
            x_middle, y_middle = lmlist[12][1:]
            x_ring, y_ring = lmlist[16][1:]
            x_nulPunt, y_nulPunt = lmlist[0][1:]
            fingersUp = detector.fingersUp(lmlist)

            if check_for_circle(fingersUp):
                
                x_previous,y_previous = 0,0
                distance = determineDistance(lmlist)
                a = distanceBalls(distance)
                b = distance*3
                y_value = int(y_nulPunt-b)
                c = 2*a
                diameter = ballGetsBigger(distance)
                drawColor = check_the_color([x_nulPunt-3*a,x_nulPunt-a,x_nulPunt+a,x_nulPunt+3*a],diameter,x_middle)
                cv2.circle(img,(x_middle,y_value),diameter-5,drawColor,-1)

                cv2.circle(img,(x_nulPunt-3*a,y_value),diameter,drawColor,2)
                
                cv2.circle(img,(x_nulPunt-a,y_value),diameter,drawColor,2)
                
                cv2.circle(img,(x_nulPunt+a,y_value),diameter,drawColor,2)
                
                cv2.circle(img,(x_nulPunt+3*a,y_value),diameter,drawColor,2)
               
            
            elif check_index_finger_up(fingersUp):
                
                if x_previous == 0 and y_previous ==0:
                    x_previous, y_previous = x_index,y_index

                brush_thickness = determineBrushThickness(lmlist) 
                
                cv2.line(img,(x_previous,y_previous),(x_index,y_index),drawColor,brush_thickness)
                cv2.line(img_previous,(x_previous,y_previous),(x_index,y_index),drawColor,brush_thickness)

                x_previous,y_previous = x_index,y_index
            
            elif pinky_is_up(fingersUp):
                x_previous,y_previous = 0,0
                img_previous = np.zeros((720,1280,3),np.uint8)

            else:
                x_previous,y_previous = 0,0

        
        '''imgGray = cv2.cvtColor(img_previous,cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,imgInv)'''
        

        img = cv2.addWeighted(img,1,img_previous,0.5,0)

        #fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime 

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),2)

        cv2.imshow("Image",img)
        
        cv2.waitKey(1)




if __name__ == '__main__':
    
    main()