import cv2
import numpy as np


def reduce_colors(image, k):
    data = np.reshape(image, (-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    data = np.float32(data)
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

    new_image = centers[labels].reshape(image.shape)
    min_val = np.min(new_image)
    span = np.max(new_image) - min_val
    new_image = (((new_image - min_val) / span * 255 + 0.5).astype(np.int32)).astype(np.uint8)
    print(new_image < 0)
    return new_image


if __name__ == '__main__':
    image_name = "cube_1_1"
    image = cv2.imread(f"./data/{image_name}.png")

    new_image = reduce_colors(image, 10)
    cv2.imshow("image", image)
    cv2.imshow("new", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
