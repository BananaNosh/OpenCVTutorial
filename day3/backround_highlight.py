import numpy as np
import cv2
from collections import deque


def highlight_back(image, new_color=(0, 0, 0), decreasing_factor=1):
    width, height = image.shape[:2]
    pixels = deque([(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)])
    already_seen = set(pixels)
    threshold = 220
    while len(pixels) > 0:
        pixel = pixels.popleft()
        value = image[pixel].copy()
        image[pixel] = new_color
        for shift in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour = pixel[0] + shift[0], pixel[1] + shift[1]
            if neighbour in already_seen \
                    or neighbour[0] < 0 or neighbour[0] >= width \
                    or neighbour[1] < 0 or neighbour[1] >= height:
                continue
            if np.mean(np.abs(image[neighbour] - value)) < threshold:
                pixels.append(neighbour)
                already_seen.add(neighbour)
    return image


if __name__ == '__main__':
    image = cv2.imread("/home/jonathan/Bilder/IMG-20190122-WA0013.jpg")
    new_image = highlight_back(image)
    cv2.imshow("new", new_image)
    cv2.waitKey()
