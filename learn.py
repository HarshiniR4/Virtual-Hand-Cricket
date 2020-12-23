import cv2
import numpy as np
import math
import random
cap = cv2.VideoCapture(0)

input_no=0
n=1
def num_gen():
    input_no=random.randint(0, 5)
    print("NUMBER TO SHOW: {}".format(input_no))
    return input_no
input_no= num_gen()
while(n):
    

    try:
        
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        
       
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:500, 100:500]
        
        cv2.rectangle(frame,(100,100),(500,500),(0,255,0),0)    
        cv2.putText(frame,"Put hand in the box", (90,90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0))
        cv2.putText(frame,"Number to show: {}".format(str(input_no)), (120,120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0))

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # define range of skin color in HSV
        lower_skin = np.array([0, 20, 80], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
        #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
        #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
                #find contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
       #find contour of max area(hand)
        if len(contours) > 0:
            #find contour of max area(hand)
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
        # cv2.drawContours(frame, cnt, -1, (0,0,255), 3)
    
        #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
        #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
        #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
        #print(arearatio)
    
    
        #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        #print(defects)
    
        #sleep(2)
        # l = no. of defects
        count_defect=0
        
        
        #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            #print(angle)
            
        
            # ignore angles > 90 
            if angle <= 90:
                count_defect += 1
                cv2.circle(roi, far, 5, (255,0,0), -1)
            
            #draw lines around hand
            cv2.line(roi,start, end, (0,255,0), 3)
            
            
        count_defect += 1
        if  cv2.waitKey(50) & 0xFF==ord('h'):
            
             
             #print corresponding gestures which are in their ranges
             font = cv2.FONT_HERSHEY_SIMPLEX
     
             if count_defect==1 :
                 cv2.putText(frame,'1',(450,120), font, 1.5, (0,0,255), 3)
                         
             elif count_defect==2:
                 cv2.putText(frame,'2',(450,120), font, 1.5, (0,0,255), 3)
                 
             elif count_defect==3:
                 cv2.putText(frame,'3',(450,120), font, 1.5, (0,0,255), 3)
                             
             elif count_defect==4:
                 cv2.putText(frame,'4',(450,120), font, 1.5, (0,0,255), 3)
                 
             elif count_defect==5:
                 cv2.putText(frame,'5',(450,120), font, 1.5, (0,0,255), 3)
             else:
                 cv2.putText(frame,'0',(450,120), font, 1.5, (0,0,255), 3)   
             if count_defect==input_no:
                 cv2.putText(frame,'CORRECT ANSWER!',(150,400), font, 1.5, (0,0,255), 3)
                 n=1
             else:
                 cv2.putText(frame,'SORRY TRY AGAIN!',(150,430), font, 1.5, (0,0,255), 3)
                 n=0

    except:
        pass             
    cv2.imshow('Gesture', frame)
    
    k = cv2.waitKey(50) & 0xFF
    if k == 27:
        break  
    
cap.release()
cv2.destroyAllWindows()