import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

TEST = 0
TRAIN = 1
SUPP = 2
setMap = ["test", "train", "supp"]


def Preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def Canny(img):
    return cv2.Canny(img, 100, 200)


def RegionOfInterest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    lowerLeftPoint = [0, 720]
    lowerRightPoint = [1280, 720]
    upperLeftPoint = [120, 360]
    upperRightPoint = [1100, 360]

    vertices = np.array([[lowerLeftPoint, upperLeftPoint,
                          upperRightPoint, lowerRightPoint]], dtype=np.int32)

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with
    # depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    try:
        if lines.any():
            for line in lines:
                if line.any():
                    for x1, y1, x2, y2 in line:
                        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    except:
        pass


def draw_lines2(img, lines, color=[255, 0, 0], thickness=2):
    bLeftValues = []  # b of left lines
    bRightValues = []  # b of Right lines
    mPositiveValues = []  # m of Left lines
    mNegitiveValues = []  # m of Right lines

    try:
        if lines.any():
            for line in lines:
                if line.any():
                    for x1, y1, x2, y2 in line:
                        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

                        # calculate slope and intercept
                        m = (y2-y1)/(x2-x1)
                        b = y1 - x1*m

                        # threshold to check for outliers
                        if m >= 0 and (m < 0.2 or m > 0.8):
                            continue
                        elif m < 0 and (m < -0.8 or m > -0.2):
                            continue

                        # seperate positive line and negative line slopes
                        if m > 0:
                            mPositiveValues.append(m)
                            bLeftValues.append(b)
                        else:
                            mNegitiveValues.append(m)
                            bRightValues.append(b)
    except:
        pass

    # Get image shape and define y region of interest value
    imshape = img.shape
    y_max = imshape[0]  # lines initial point at bottom of image
    y_min = 350        # lines end point at top of ROI

    # Get the mean of all the lines values
    AvgPositiveM = mean(mPositiveValues)
    AvgNegitiveM = mean(mNegitiveValues)
    AvgLeftB = mean(bLeftValues)
    AvgRightB = mean(bRightValues)

    # use average slopes to generate line using ROI endpoints
    if AvgPositiveM != 0:
        x1_Left = (y_max - AvgLeftB)/AvgPositiveM
        y1_Left = y_max
        x2_Left = (y_min - AvgLeftB)/AvgPositiveM
        y2_Left = y_min

    if AvgNegitiveM != 0:
        x1_Right = (y_max - AvgRightB)/AvgNegitiveM
        y1_Right = y_max
        x2_Right = (y_min - AvgRightB)/AvgNegitiveM
        y2_Right = y_min

        # define average left and right lines
        try:
            cv2.line(img, (int(x1_Left), int(y1_Left)), (int(x2_Left),
                                                         int(y2_Left)), color, thickness)  # avg Left Line
            cv2.line(img, (int(x1_Right), int(y1_Right)), (int(
                x2_Right), int(y2_Right)), color, thickness)  # avg RightLine
        except:
            pass


def mean(list):
    """
    calculate mean of list
    """
    return float(sum(list)) / max(len(list), 1)


def hough_lines(img):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_len = 20
    max_line_gap = 20

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]))

    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)

    draw_lines(line_img, lines)
    return line_img


def getImage(n, set):
    path = "benchmark_velocity_supp/supp_img/" + str(n).zfill(4) + ".jpg"
    if (set == TEST or set == TRAIN):
        path = "benchmark_velocity_" + \
            setMap[set] + "/clips/" + str(n) + "/imgs/040.jpg"

    print(path)
    img = cv2.imread(path)
    return img


def main():
    for i in range(1, 100):
        img = getImage(i, TRAIN)
        preprocessedImg = Preprocess(img)
        cannyImg = Canny(preprocessedImg)
        maskedImg = RegionOfInterest(cannyImg)
        houged = hough_lines(maskedImg)

        if (not os.path.exists('output')):
            print('creating output dir')
            os.mkdir('./output')

        cv2.imwrite("output/output" + str(i) + ".jpg", RegionOfInterest(img))


main()
