from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from imutils.video import FileVideoStream
from imutils.video import VideoStream


def faces_for_image(image):
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for i, rect in enumerate(rects):
        # noinspection PyRedeclaration
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        x, y, w, h = face_utils.rect_to_bb(rect)
        stroke_color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), stroke_color, 2)

        cv2.putText(image, f"Face {i+1}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, stroke_color)

        for x, y in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    return image


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", help="path to input image")
ap.add_argument("-v", "--video", help="path to input video")

args = vars(ap.parse_args())
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(args["shape_predictor"])

image_path = args.get("image", False)
if image_path:
    image = cv2.imread(args["image"])
    image = faces_for_image(image)
    cv2.imshow("Output", image)
    cv2.waitKey(0)

else:
    video_path = args.get("video", False)
    fileStream = bool(video_path)
    if not video_path:
        vs = VideoStream(src=0).start()
    else:
        vs = FileVideoStream(video_path).start()

    while True:
        if fileStream and not vs.more():
            break

        frame = vs.read()
        frame = faces_for_image(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
