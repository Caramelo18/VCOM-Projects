import cv2
import sys
import numpy as np
import math


def preprocessing(img):
    blurred = cv2.bilateralFilter(img, d=9, sigmaColor=100, sigmaSpace=100)
    return blurred

def skin_color_segmentation(hsv):
    # define range of skin color in HSV
    lower_skin = np.array([1,45,0], dtype=np.uint8)
    upper_skin = np.array([15,200,255], dtype=np.uint8)

    # extract skin color mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    return mask

def apply_morph_operations(img):
    # extrapolate the hand to fill dark spots within it
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 3)
    img = cv2.dilate(img, kernel, iterations = 2)
    return img

def find_hand_contour(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # select only contour of object with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    return cnt

def process_image(img):
    img = preprocessing(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = skin_color_segmentation(hsv)

    mask = apply_morph_operations(mask)
    cv2.imshow("Mask", mask)

    cnt = find_hand_contour(mask)
    
    # find defects in convex hull of hand
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    # l = no. of defects
    l = 0

    max_d = 0

    height, _, _ = img.shape

    #code for finding no. of defects due to fingers
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])


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

        #print(angle, d)

        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)

        ratio = height / d
        if angle <= 90 and ratio < 8:
            l += 1
            cv2.circle(img, far, 5, [255,0,0], -1)

        if d > max_d:
            max_d = d

        #draw lines around hand
        cv2.line(img,start, far, [0,255,0], 2)
        cv2.line(img,far, end, [0,255,0], 2)


        cv2.circle(img, far, 2, [0,0,255], -1)

    ratio = height / max_d
    if ratio < 10:
        l += 1

    print("Number of fingers: ", l)

    cv2.imshow('Mask', mask)

    cv2.imshow('Hand Mask', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def capture_image():
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('frame')
    cv2.resizeWindow('frame', 634, 357)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = cv2.resize(frame, (634, 357), cv2.INTER_AREA)

        # Display the resulting frame
        cv2.imshow('frame',frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('d'):
            process_image(frame)

    cap.release()
    cv2.destroyAllWindows()

def main(argv):
    if len(argv) is 0:
        capture_image()
    elif len(argv) is 1:
        file = argv[0]
        path = "samples/" + file

        img = cv2.imread(path, 1)
        process_image(img)


if __name__ == '__main__':
    main(sys.argv[1:])
