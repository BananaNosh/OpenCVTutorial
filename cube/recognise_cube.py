import imutils
from imutils.perspective import four_point_transform
from imutils.perspective import order_points
import cv2
import numpy as np
import math


def transform_according_to_reference_square(image, reference_points):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(reference_points)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    new_ref_width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    new_ref_width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    new_ref_width = max(int(new_ref_width_1), int(new_ref_width_2))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    new_ref_height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    new_ref_height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    new_ref_height = max(int(new_ref_height_1), int(new_ref_height_2))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        rect[0],
        rect[0] + [new_ref_width - 1, 0],
        rect[0] + [new_ref_width - 1, new_ref_height - 1],
        rect[0] + [0, new_ref_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    # return the warped image
    return warped


def find_colored_squares_in_image(image):
    image = imutils.resize(image, height=500)
    print(np.std(image, axis=0))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 100)
    print("STEP 1: Edge Detection")

    rectangles = get_recs(edged, colored_image=image)
    if len(rectangles) == 0:
        return image

    warp_ratios = get_warp_ratios(rectangles)

    warp_ratio = np.mean(warp_ratios)
    if False and warp_ratio > 0.1:
        ref_index = np.argmin(np.abs(warp_ratios - warp_ratio))
        reference_rectangle = rectangles[ref_index]
        image = transform_according_to_reference_square(image, np.reshape(reference_rectangle, (4, 2)))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 100)

        rectangles, _ = get_recs(edged, 0.1)

    rectangles = list(rectangles)

    if len(rectangles) < 9:
        for i, (lower, upper) in enumerate(zip([np.array([0, 0, 70]), np.array([78, 0, 0])],
                                               [np.array([91, 255, 145]), np.array([169, 255, 42])])):
            only_colored = cv2.inRange(image, lower, upper)
            colored_recs = get_recs(only_colored, rel_similarity_threshold=0.2, already_found_recs=rectangles)
            if len(colored_recs):
                rectangles.extend(colored_recs)
                rectangles = list(remove_doubles(rectangles))

        # green_mask = np.zeros_like(image)
        # green_mask[:, :, 1] = 100
        # only_colored = cv2.bitwise_and(green_mask, green_mask, mask=only_colored)
        # shifted_green = cv2.add(image, only_colored)
        # # shifted_green[np.where(shifted_green[:, :, 1] < 110)] += np.array([0, 100, 0])
        # cv2.imshow("shifted", shifted_green)

    # ordered_corners = order_points(np.reshape(approx, (4, 2)))
    # sq_width = ordered_corners[1][0] - ordered_corners[0][0]
    # sq_height = ordered_corners[3][1] - ordered_corners[0][1]
    # print(sq_width, sq_height)
    # print(ordered_corners[0])
    # rectangles.append(approx)
    print(f"found {len(rectangles)}")
    cv2.drawContours(image, rectangles, -1, (0, 255, 0), 2)
    return image


def get_warp_ratios(rectangles):
    np_rectangles = np.zeros((len(rectangles), 4, 1, 2))
    for i, rect in enumerate(rectangles):
        np_rectangles[i] = rect
    # print(np_rectangles.shape, rectangles.shape)
    flattened_recs = np_rectangles[:, :, 0]
    right_order_recs = np.array([order_points(rec) for rec in flattened_recs])
    warp_ratio_dist = right_order_recs[:, 3] - right_order_recs[:, 0]
    warp_ratios = np.abs(warp_ratio_dist[:, 0] / warp_ratio_dist[:, 1])
    return warp_ratios


def get_recs(image_to_extract_from, rel_similarity_threshold=0.05, colored_image=None, already_found_recs=None):
    cnts = cv2.findContours(image_to_extract_from.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rectangles = []
    if already_found_recs is not None:
        for rec in already_found_recs:
            rectangles.append(np.array([rec, cv2.contourArea(rec)]))
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found a square
        if len(approx) == 4:
            sq_area = cv2.contourArea(approx)
            # print(sq_area)
            if sq_area < 1000:
                break

            # check for similarity in color within approx
            if colored_image is not None:
                x_min, y_min = np.min(approx, axis=(0, 1))
                x_max, y_max = np.max(approx, axis=(0, 1))
                width = x_max - x_min
                height = y_max - y_min
                x_min = int(x_min + 0.1 * width)
                x_max = int(x_max - 0.1 * width)
                y_min = int(y_min + 0.1 * height)
                y_max = int(y_max - 0.1 * height)
                mean_color_std = np.mean(np.std(colored_image[y_min:y_max, x_min:x_max], axis=(0, 1)))
                print("std", mean_color_std)
                if mean_color_std > 15:
                    continue
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # check for similarity in size
            if len(rectangles) > 0 \
                    and not math.isclose(sq_area, rectangles[-1][1], rel_tol=rel_similarity_threshold):
                if len(rectangles) < 3:
                    rectangles = []
                else:
                    continue
            rectangles.append(np.array([approx, sq_area]))
    if len(rectangles) == 0:
        return np.array([])
    rectangles = np.array(rectangles)
    rectangles = rectangles[:, 0]
    # remove doubles
    rectangles = sorted(rectangles, key=lambda x: np.min(np.sum(x[:, 0], axis=1)))
    rectangles = remove_doubles(rectangles)
    return rectangles


def remove_doubles(rectangles):
    for i, rec in enumerate(rectangles[:-1]):
        for neighbour in rectangles[i + 1:]:
            print(np.sum(np.abs(rec[np.argmin(np.sum(rec[:, 0], axis=1)), 0]
                                - neighbour[np.argmin(np.sum(neighbour[:, 0], axis=1)), 0])))
            if np.sum(np.abs(rec[np.argmin(np.sum(rec[:, 0], axis=1)), 0]
                             - neighbour[np.argmin(np.sum(neighbour[:, 0], axis=1)), 0])) < 10:
                rectangles[i] = None
                break
    rectangles = np.array([rec for rec in rectangles if rec is not None])
    return rectangles


if __name__ == '__main__':
    # images_to_print = [f"cube_1_{i}" for i in range(6)]
    images_to_print = ["cube_1_1"]
    # images_to_print = [f"cube_1_{i}_warped" for i in range(3, 6)]
    for image_name in images_to_print:
        image = cv2.imread(f"./data/{image_name}.png")
        image = find_colored_squares_in_image(image)
        cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
