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

def count_finger_defects(defects, max_length, cnt, img):
    finger_defects_count = 0
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
        ratio = defect_length / max_length
        if angle <= 90 and ratio > 0.15:
            finger_defects_count += 1
            cv2.circle(img, far, 5, [255, 0, 0], -1)

        # draw lines around hand
        cv2.line(img, start, far, [0,255,0], 2)
        cv2.line(img, far, end, [0,255,0], 2)

        # draw defect point
        cv2.circle(img, far, 2, [0,0,255], -1)

    return finger_defects_count


def get_triangle_area(a, b, c):
    # Given the length of each side of the triangle, returns its area using Heron's formula
    perimeter = (a+b+c)/2
    return math.sqrt(perimeter * (perimeter - a) * (perimeter - b) * (perimeter - c))

def cnt_size(cnt):
    _, _, w, h = cv2.boundingRect(cnt)
    return max(w, h)

def get_bounding_rectangle_area(cnt):
    _, _, w, h = cv2.boundingRect(cnt)
    return w * h

def process_image(img, show_flag=False):
    img = preprocessing(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = skin_color_segmentation(hsv)

    mask = apply_morph_operations(mask)

    cnt = find_hand_contour(mask)
    defects = find_convexity_defects(cnt)
    max_length = cnt_size(cnt)

    finger_defects_count = count_finger_defects(defects, max_length, cnt, img)

    # if no finger defects are found, checks if contour area is > 65% of bounding rectangle area, as that indicates a closed hand
    if finger_defects_count == 0:
        cnt_area = cv2.contourArea(cnt)
        bounding_rectangle_area = get_bounding_rectangle_area(cnt)
        area_ratio = cnt_area / bounding_rectangle_area
        if area_ratio > 0.65:
            finger_count = 0
        else:
            finger_count = 1
    else:
        finger_count = finger_defects_count + 1

    if show_flag:
        print("Number of fingers: ", finger_count)
        cv2.imshow('Mask', mask)
        cv2.imshow('Hand', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return finger_count


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
            process_image(frame, show_flag=True)

    cap.release()
    cv2.destroyAllWindows()

def run_tests():
    tests = {
        '0_0.jpg': 0,
        '0.jpg': 0,
        '0cama.jpg': 0,
        '0dark.jpg': 0,
        '0fake.jpg': 0,
        '1_lado.jpg': 1,
        '1.jpg': 1,
        '1cama.jpg': 1,
        '1background.jpg': 1,
        '2.jpg': 2,
        '2far.jpg': 2,
        '2far1.jpg': 2,
        '2dark.jpg': 2,
        '2fake.png': 2,
        '2upside.jpg': 2,
        '3.jpg': 3,
        '3far.jpg': 3,
        '3obs.jpg': 3,
        '3side.jpg': 3,
        '31.jpg': 3,
        '4.jpg': 4,
        '4side.jpg': 4,
        '4diagonal.jpg': 4,
        '5.jpg': 5,
        '5side.jpg': 5,
        'diogo1.jpg': 1,
        'diogo3.jpg': 3,
        'diogo4.jpg': 4,
        'diogo42.jpg': 4,
        'fabio5.jpg': 5,
        'img1.png': 1,
        'img2.png': 2,
        'img3.png': 3,
        'img4.png': 4,
        'img5.png': 5
    }

    for test in tests:
        path = "samples/" + test
        img = cv2.imread(path, 1)
        result = process_image(img)
        if result == tests[test]:
            print('{}: OK'.format(test))
        else:
            print('{}: Expected {} but received {}.'.format(test, tests[test], result))

def main(argv):
    if len(argv) is 0:
        capture_image()
    elif len(argv) is 1:
        if argv[0] == 'tests':
            run_tests()
        else:
            file = argv[0]
            path = "samples/" + file
            img = cv2.imread(path, 1)
            process_image(img, show_flag=True)


if __name__ == '__main__':
    main(sys.argv[1:])
