import cv2
import sys
import numpy as np
import math

def process_image(path):
    img = cv2.imread(path, 1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((1,1),np.uint8)
    cv2.imshow("HSV", hsv)
    # define range of skin color in HSV
    lower_skin = np.array([0,20,50], dtype=np.uint8)
    upper_skin = np.array([50,255,255], dtype=np.uint8)

    #extract skin colur imagw
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    #extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask,kernel,iterations = 3)

    #blur the image
    mask = cv2.GaussianBlur(mask,(3,3),500)
    cv2.imshow("Mask", mask)

    #find contours
    _,contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #find contour of max area(hand)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

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

    #find the defects in convex hull with respect to hand
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    # l = no. of defects
    l=0

    #code for finding no. of defects due to fingers
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt= (100,180)


        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        s = (a+b+c)/2
        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

        #distance between point and convex hull
        d=(2*ar)/a

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57


        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
        if angle <= 90 and d>30:
            l += 1
            cv2.circle(img, far, 3, [255,0,0], -1)

        #draw lines around hand
        cv2.line(img,start, end, [0,255,0], 2)


    l+=1


    if l==1:
        print(0)
    elif l==2:
        print(1)
    elif l==3:
        print(2)
    elif l==4:
        print(3)
    elif l==5:
        print(4)
    elif l==6:
        print(5)


    cv2.imshow('Hand',mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(argv):
    if len(argv) is not 1:
        print("usage")
        sys.exit(1)

    file = argv[0]
    path = "samples/" + file
    process_image(path)


if __name__ == '__main__':
    main(sys.argv[1:])
