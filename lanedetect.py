import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from numpy import ones, vstack
from numpy.linalg import lstsq

TEST = 0
TRAIN = 1
SUPP = 2
setMap = ["test", "train", "supp"]


def area(x1, y1, x2, y2, x3, y3):

    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)


# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def isInside(x1, y1, x2, y2, x3, y3, x, y):

    # Calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)

    # Calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)

    # Calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)

    # Calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)

    # Check if sum of A1, A2 and A3
    # is same as A
    if(A == A1 + A2 + A3):
        return True
    else:
        return False


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


def Preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def OtsuImg(img):
    ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu


def transformThreshold(img, thresholdValue):
    outputImage = np.full(img.shape, 0)

    for i in range(outputImage.shape[0]):
        for j in range(outputImage.shape[1]):
            outputImage[i, j] = 255 * (img[i, j] > thresholdValue)

    return np.uint8(outputImage)


def Canny(img):
    return cv2.Canny(img, 100, 200)


def RegionOfInterest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    lowerLeftPoint = [0, 660]
    lowerRightPoint = [1280, 660]
    upperLeftPoint = [120, 390]
    upperRightPoint = [1100, 390]

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


def hough_lines(img, drawlinesOg=False):
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

    if drawlinesOg:
        draw_lines(line_img, lines)
    else:
        sectionVal = 1280 // 20
        sections = [i for i in range(0, 1280 + 1, sectionVal)]
        print(sections)

        for i in range(len(sections) - 1):
            baseLeft, baseRight = sections[i], sections[i + 1]
            l = []
            try:
                for x in lines:
                    xVal = x[0][0]
                    yVal = x[0][1]
                    if isInside(baseLeft, 720, baseRight, 720, 640, 0, xVal, yVal):
                        # print(x[0])
                        l.append(x[0])
            except:
                pass

            draw_lines2(line_img, l)

    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


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
    """
    This function draws `lines` with `color` and `thickness`.    
    """
    imshape = img.shape

    # these variables represent the y-axis coordinates to which
    # the line will be extrapolated to
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]

    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []

    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    #print("lines", lines)

    for line in lines:
        # print(line)
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]

        gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        ymin_global = min(min(y1, y2), ymin_global)
        #print("gradient for", line, gradient)

        if (gradient > 0):
            all_left_grad += [gradient]
            all_left_y += [y1, y2]
            all_left_x += [x1, x2]
        else:
            all_right_grad += [gradient]
            all_right_y += [y1, y2]
            all_right_x += [x1, x2]

    #print("all left gradients", all_left_grad)

    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

    #print("all right gradients", all_right_grad)

    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)

    # Make sure we have some points in each lane line category
    if (len(all_left_grad) > 0):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)

        # print('left')
        #print(upper_left_x, ymin_global, lower_left_x, ymax_global)
        try:
            cv2.line(img, (upper_left_x, ymin_global),
                     (lower_left_x, ymax_global), color, thickness)
        except:
            pass

    if (len(all_right_grad) > 0):
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)
        # print('right')
        #print(upper_right_x, ymin_global, lower_right_x, ymax_global)
        try:
            print('drawing line right')
            cv2.line(img, (upper_right_x, ymin_global),
                     (lower_right_x, ymax_global), color, thickness)
        except:
            pass


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
        print(i)
        img = getImage(i, TRAIN)
        cv2.imwrite("output/output" + str(i) +
                    "Original.jpg", RegionOfInterest(img))

        preprocessedImg = Preprocess(img)
        cannyImg = Canny(np.uint8(preprocessedImg))
        maskedImg = RegionOfInterest(cannyImg)
        houged = hough_lines(maskedImg)
        houged2 = hough_lines(maskedImg, True)
        weighted = weighted_img(houged, img)
        weighted2 = weighted_img(houged2, img)

        if (not os.path.exists('output')):
            print('creating output dir')
            os.mkdir('./output')

        cv2.imwrite("output/output" + str(i) + ".jpg", np.uint8(weighted))
        cv2.imwrite("output/output" + str(i) +
                    "DrawOriginal.jpg", np.uint8(weighted2))


main()
