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
