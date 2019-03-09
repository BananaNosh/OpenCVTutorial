import cv2
import numpy as np


def stack_up_frames(video_file):
    capture = cv2.VideoCapture(video_file)
    frames = []
    border = None
    while True:
        _, frame = capture.read()
        if frame is None:
            break
        if border is None:
            border = cv2.inRange(frame, np.array([0, 50, 200]), np.array([50, 150, 255]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)
        frames.append(frame)
    stack_ups = []
    parts_count = 5
    for i in range(parts_count):
        stack_up = np.zeros_like(frames[0])
        part_length = (len(frames)-1) // parts_count
        for frame in frames[i*part_length: (i+1)*part_length]:
            cv2.imshow("video", frame)
            # cv2.waitKey(20)
            stack_up = np.where(frame > stack_up, frame, stack_up)
            stack_ups.append(stack_up)
        imp_area_cont = cv2.findContours(border, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
        # cv2.drawContours(border, imp_area_cont[0:1], -1, 127, 2)
        x_min, y_min= np.min(imp_area_cont[0], axis=(0, 1))
        x_max, y_max= np.max(imp_area_cont[0], axis=(0, 1))
        cv2.rectangle(border, (x_min, y_min), (x_max, y_max), 200, thickness=2)
        zoomed = np.full_like(stack_up, 255)
        zoomed[y_min:y_max, x_min:x_max] = stack_up[y_min:y_max, x_min:x_max]
        # cv2.fillPoly(border, pts=imp_area_cont, color=(255,255,255))

        # contours = cv2.findContours(stack_up, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
        # shapes = []
        # for c in contours:
        #     # approximate the contour
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        #     if cv2.contourArea(approx) > 20:
        #         shapes.append(approx)
        #
        # with_cons = stack_up.copy()
        # cons = np.zeros_like(stack_up)
        # cv2.drawContours(with_cons, shapes, -1, 127, 2)
        # cv2.drawContours(cons, shapes, -1, 255, 2)
        cv2.imshow(f"stack_{i}", stack_up)
        # cv2.imshow(f"zoomed_{i}", zoomed)
        # cv2.imshow("cons and ", with_cons)
        # cv2.imshow("cons", cons)
        # cv2.imshow("border", border)
    cv2.waitKey(0)


if __name__ == '__main__':
    # filename = "az_45004e5ac71c40749add8ae3abaa2db5"
    filename = "az_f525e35e4e3b46bfba27f6b600e1ddc5"
    vide_file = f"./data/learning/stalled/{filename}.mp4"
    # stack_up_frames(vide_file)
    capture = cv2.VideoCapture(vide_file)
    for i in range(26):
        capture.read()
    _, frame = capture.read()
    cv2.imshow("fr", frame)
    cv2.waitKey(0)
    # 463 311  instead of 888 422
    # 218 310  instead of 904 422
    # 370 184  instead of 804 295
