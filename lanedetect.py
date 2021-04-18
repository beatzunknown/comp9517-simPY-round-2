import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
from numpy import ones, vstack
from numpy.linalg import lstsq

COLOR = [255, 0, 0]
THICKNESS = 2

def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)

# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def is_inside(x1, y1, x2, y2, x3, y3, x, y):

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

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)

def canny(img):
    return cv2.Canny(img, 100, 200)

def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    bot_left = [0, 720]
    bot_right = [1280, 720]
    top_left = [120, 360]
    top_right = [1100, 360]

    vertices = np.array([[bot_left, top_left,
                          top_right, bot_right]], dtype=np.int32)

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

# function no longer used
def otsu_img(img):
    ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu

# function no longer used
def transform_threshold(img, threshold_value):
    output_image = np.full(img.shape, 0)

    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            output_image[i, j] = 255 * (img[i, j] > threshold_value)

    return np.uint8(output_image)


def hough_lines(img, line_type):
    """
    `img` should be the output of a canny transform.

    Returns an image with hough lines drawn.
    """
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_len = 20
    max_line_gap = 20

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]))
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)

    lanes = None
    h_samples = None

    if line_type == 'final':
        lanes, h_samples = filter_and_draw_lines(line_img, lines)
    elif line_type == 'hough':
        draw_lines(line_img, lines)
    elif line_type == 'extrapolated':
        draw_lines(line_img, lines, True)
    elif line_type == 'filtered':
        draw_lines(line_img, lines, True, True)

    return line_img, lanes, h_samples

def extrapolate_line(img, gradient, x1, y1, filtered):
    c = y1 - (gradient * x1)
    top_val = 360
    top_point = (top_val - c) // gradient
    bot_point = (720 - c) // gradient

    y_bot_point = 720
    y_top_point = top_val

    if bot_point < 0:
        bot_point = 0
        y_bot_point = c
    elif bot_point > 1280:
        bot_point = 1280
        y_bot_point = (gradient * bot_point) + c

    if top_point < 0:
        top_point = 0
        y_top_point = c
    elif top_point > 1280:
        top_point = 1280
        y_top_point = (gradient * top_point) + c

    if (int(top_point) in range(600, 750)) and filtered:
        return [int(bot_point), int(y_bot_point), int(top_point), int(y_top_point)]
    elif not filtered:
        return [int(bot_point), int(y_bot_point), int(top_point), int(y_top_point)]

def get_edge_lines(lines_list):
    edge_lines = []
    for i in range(len(lines_list)):
        lines = lines_list[i]
        axis = int(i % 3 == 0)

        if len(lines) > 0:
            currMax = lines[0][axis]
            index = 0
            for i in range(len(lines)):
                line = lines[i]
                if line[axis] > currMax:
                    index = i
                    currMax = line[axis]

            edge_lines.append(lines[index])
        else:
            edge_lines.append(None)
        
    return edge_lines

def filter_and_draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    filtered_lines = []
    try:
        for line in lines:
            if line.any():
                for x1, y1, x2, y2 in line:
                    gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
                    filtered_lines.append(extrapolate_line(img, gradient, x1, y1, True))
    except:
        pass

    left_lines = [l for l in filtered_lines if l and l[0] == 0]
    right_lines = [l for l in filtered_lines if l and l[0] == 1280]
    bot_lines = [l for l in filtered_lines if l and 0 < l[0] < 1280]

    bot_left_filtered_lines = []
    bot_right_filtered_lines = []
    left_filtered_lines = []
    right_filtered_lines = []

    # for the lines on the bottom
    for i in range(0, 1152 + 1, 128):
        valid_lines = [l for l in bot_lines if i <= l[0] < i+128 and l[1] == 720]

        if valid_lines:
            valid_line = [int(coord) for coord in np.median(valid_lines, 0)]
            if valid_line[0] < 640:
                bot_left_filtered_lines.append(valid_line)
            else:
                bot_right_filtered_lines.append(valid_line)

    # lines on the left and the right sides
    for i in range(432, 576 + 1, 144):
        valid_left_lines = [l for l in left_lines if i <= l[1] < i+144 and 450 < l[1] < 600]
        valid_right_lines = [l for l in right_lines if i <= l[1] < i+144 and 450 < l[1] < 600]

        if valid_left_lines:
            valid_line = [int(coord) for coord in np.median(valid_left_lines, 0)]
            left_filtered_lines.append(valid_line)

        if valid_right_lines:
            valid_line = [int(coord) for coord in np.median(valid_right_lines, 0)]
            right_filtered_lines.append(valid_line)

    filtered_lines = [left_filtered_lines, bot_left_filtered_lines, bot_right_filtered_lines, right_filtered_lines]
    edge_lines = get_edge_lines(filtered_lines)

    for edge_line in edge_lines:
        if edge_line:
            cv2.line(img, (int(edge_line[0]), int(edge_line[1])),
                    (int(edge_line[2]), int(edge_line[3])), COLOR, 10)


    lanes, h_samples = formatData(edge_lines)
    print('there are', len(lanes), 'lanes in image')
    return lanes, h_samples

def draw_lines(img, lines, extrapolated=False, filtered=False):
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if extrapolated:
                gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
                l = extrapolate_line(img, gradient, x1, y1, filtered)
                if l:
                    cv2.line(img, (l[0], l[1]), (l[2], l[3]), COLOR, THICKNESS)
            else:
                cv2.line(img, (x1, y1), (x2, y2), COLOR, THICKNESS)
    except:
        pass

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def formatData(lines):
    # print(lines)
    h_samples = [i for i in range(240, 711, 10)]
    # print(h_samples)
    lanes = []

    for l in lines:
        # print(l)
        if l != None:
            x1, y1, x2, y2 = l
            gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            # y = mx + c
            c = y1 - (gradient * x1)
            lane = []

            for y in h_samples:
                if y < 360:
                    lane.append(-2)
                else:
                    x_val = (y - c) // gradient
                    lane.append(int(x_val))

            lanes.append(lane)

    return lanes, h_samples

def main():

    # Uncomment to use lane detection dataset

    # label_file = 'train_set/label_data_0313.json'
    # with open(label_file) as f:
    #     labels = json.load(f)

    for i in range(11, 12):
        # Choose which dataset to use
        path = f'benchmark_velocity_train/clips/{i+1}/imgs/040.jpg'
        # path = f"train_set/{labels[i]['raw_file']}"
        img = cv2.imread(path)

        preprocessed_img = preprocess(img)
        canny_img = canny(np.uint8(preprocessed_img))
        masked_img = region_of_interest(canny_img)

        img_hough_lines = weighted_img(hough_lines(masked_img, 'hough')[0], img)
        img_extrapolated = weighted_img(hough_lines(masked_img, 'extrapolated')[0], img)
        img_filtered = weighted_img(hough_lines(masked_img, 'filtered')[0], img)

        img_final, lanes, h_samples = hough_lines(masked_img, 'final')
        img_final = weighted_img(img_final, img)

        for l in lanes:
            print(l)

        print(h_samples)

        if not os.path.exists('output'):
            print('creating output dir')
            os.mkdir('./output')

        cv2.imwrite(f'output/output{str(i).zfill(3)}Preprocess.jpg', preprocessed_img)
        cv2.imwrite(f'output/output{str(i).zfill(3)}Canny.jpg', canny_img)
        cv2.imwrite(f'output/output{str(i).zfill(3)}ROI.jpg', masked_img)

        cv2.imwrite(f'output/output{str(i).zfill(3)}Hough.jpg', np.uint8(img_hough_lines))
        cv2.imwrite(f'output/output{str(i).zfill(3)}Extrapolated.jpg', np.uint8(img_extrapolated))
        cv2.imwrite(f'output/output{str(i).zfill(3)}Filtered.jpg', np.uint8(img_filtered))

        cv2.imwrite(f'output/output{str(i).zfill(3)}Final.jpg', np.uint8(img_final))

'''
{
  "lanes": [
        [-2, -2, -2, -2, 632, 625, 617, 609, 601, 594, 586, 578, 570, 563, 555, 547, 539, 532, 524, 516, 508, 501, 493, 485, 477,
            469, 462, 454, 446, 438, 431, 423, 415, 407, 400, 392, 384, 376, 369, 361, 353, 345, 338, 330, 322, 314, 307, 299],
        [-2, -2, -2, -2, 719, 734, 748, 762, 777, 791, 805, 820, 834, 848, 863, 877, 891, 906, 920, 934, 949, 963, 978, 992, 1006, 1021,
            1035, 1049, 1064, 1078, 1092, 1107, 1121, 1135, 1150, 1164, 1178, 1193, 1207, 1221, 1236, 1250, 1265, -2, -2, -2, -2, -2],
        [-2, -2, -2, -2, -2, 532, 503, 474, 445, 416, 387, 358, 329, 300, 271, 241, 212, 183, 154, 125, 96, 67,
            38, 9, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        [-2, -2, -2, 781, 822, 862, 903, 944, 984, 1025, 1066, 1107, 1147, 1188, 1229, 1269, -2, -2, -2, -2, -2, - \
            2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]
       ],
  "h_samples": [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710],
  "raw_file": "path_to_clip"
}
'''

main()
