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
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 8)
    return img

def find_hand_contour(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # select only contour of object with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    return cnt

def find_convexity_defects(contour):
    # find defects in convex hull of hand
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    return defects

def get_triangle_area(a, b, c):
    # Given the length of each side of the triangle, returns its area using Heron's formula
    perimeter = (a+b+c)/2
    return math.sqrt(perimeter * (perimeter - a) * (perimeter - b) * (perimeter - c))

def cnt_size(img, cnt):
    _,_,w,h = cv2.boundingRect(cnt)
    return max(w, h)

def process_image(img):
    img = preprocessing(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = skin_color_segmentation(hsv)

    mask = apply_morph_operations(mask)
    cv2.imshow("Mask", mask)

    cnt = find_hand_contour(mask)

    defects = find_convexity_defects(cnt)

    max_length = cnt_size(img, cnt)

    finger_defects_count = 0

    max_defect_length = 0


    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # find length of all sides of the defect triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        area = get_triangle_area(a, b, c)

        # distance between defect point and convex hull (height of the defect triangle)
        defect_length = (2 * area) / a

        # find the angle of the defect triangle
        angle = math.degrees(math.acos((b**2 + c**2 - a**2 )/ (2 * b * c)))

        # ignore angles > 90 and ignore points very close to convex hull (they generally correspond to noise or imperfections in the mask)
        ratio = max_length / defect_length
        if angle <= 87 and ratio < 8:
            finger_defects_count += 1
            cv2.circle(img, far, 5, [255, 0, 0], -1)

        if defect_length > max_defect_length:
            max_defect_length = defect_length

        # draw lines around hand
        cv2.line(img,start, far, [0,255,0], 2)
        cv2.line(img,far, end, [0,255,0], 2)

        # draw defect point
        cv2.circle(img, far, 2, [0,0,255], -1)

    # find if longest defect is big enough to be considered a finger
    ratio = max_length / max_defect_length
    if ratio < 9:
        finger_defects_count += 1

    print("Number of fingers: ", finger_defects_count)

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
