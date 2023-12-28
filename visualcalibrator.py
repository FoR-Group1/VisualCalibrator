import matplotlib.pyplot as plt
import argparse
import numpy
import json
import cv2
import os

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters_create()

def aruco_id_to_global_corner_index(aruco_id):
    if aruco_id == 24:
        return 0
    elif aruco_id == 25:
        return 1
    elif aruco_id == 26:
        return 3
    elif aruco_id == 27:
        return 2

def main(frames_dir, gcode):
    plot_num = 0
    for k, im_name in enumerate(sorted(os.listdir(frames_dir)), 0):
        im_path = os.path.join(frames_dir, im_name)

        im = cv2.imread(im_path)

        (corners, ids, rejected) = cv2.aruco.detectMarkers(im, aruco_dict, parameters = aruco_params)
        # print("Detected %d markers" % len(corners))

        labelled_image = im.copy()
        cv2.aruco.drawDetectedMarkers(labelled_image, corners, ids)

        global_corners = []

        for i, id_ in enumerate(ids, 0):
            global_corners.append(corners[i][0][aruco_id_to_global_corner_index(id_[0]), :])

        for c in global_corners:
            labelled_image = cv2.circle(labelled_image, [int(i) for i in c], 5, (0, 255, 0), -1)

        global_corners = order_corners(numpy.float32(global_corners))
        dst, width, height = get_persp_size(global_corners)
        persp_m = cv2.getPerspectiveTransform(global_corners, dst)
        warped = cv2.warpPerspective(im, persp_m, (width, height))

        warped_mask = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        warped_mask = cv2.inRange(warped_mask, numpy.uint8([0, 85, 85]), numpy.uint8([179, 255, 255]))

        canny = cv2.Canny(warped_mask, 100, 200)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(warped, contours, -1, (0, 255, 0), 1)

        cv2.imshow("Original", labelled_image)
        cv2.imshow("Perspective Transform", warped)
        cv2.imshow("Perspective Transform Mask", canny)
        
        # assume all the countours that are found are of the same shape (sometimes it gets separated around the point)
        # but yeah this assumes the thresholding perfect (which it usually is)
        contour = numpy.concatenate(contours, axis = 0)
        # print(contour.shape)
        q = contour[:, 0, :]
        
        if k % 6 == 0:
            plot_num += 1

        plt.figure(plot_num)
        plt.scatter(contour[:, 0, 0], contour[:, 0, 1])
        plt.savefig(os.path.join("Plots", "Plot-%i.png" % plot_num))

        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
                   
    cv2.destroyAllWindows()

def gradient(pt_1, pt_2):
    up = pt_1[1] - pt_2[1]
    across = pt_1[0] = pt_2[0]
    return up / across

def order_corners(global_corners):
    # the order of points needs to be consistant
    # we use tl, tr, br, bl
    rect = numpy.zeros((4, 2), dtype = "float32")
    s = global_corners.sum(axis = 1)
    rect[0] = global_corners[numpy.argmin(s)]
    rect[2] = global_corners[numpy.argmax(s)]
    diff = numpy.diff(global_corners, axis = 1)
    rect[1] = global_corners[numpy.argmin(diff)]
    rect[3] = global_corners[numpy.argmax(diff)]
    return rect

def get_persp_size(global_corners):
    tl, tr, br, bl = global_corners

    # top and bottom, then get the biggest
    width_1 = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_2 = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_1), int(width_2))

    # left and right, then get the biggest
    height_1 = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_2 = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_1), int(height_2))

    return numpy.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ]), max_width, max_height


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--frames-dir",
        help = "Path to a directory containing image frames",
        required = True,
        type = os.path.abspath
    )
    parser.add_argument(
        "-g", "--gcode",
        help = "Path to a file containing timestamped gcode commands",
        required = True,
        type = os.path.abspath
    )
    return vars(parser.parse_args())

if __name__ == "__main__":
    main(**get_argparser())