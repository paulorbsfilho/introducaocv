import cv2 as cv
import numpy as np


def main():
    img = cv.imread('images_in/aviao.jpg', 0)
    cv.imshow('Original', img)
    rows, cols = img.shape

    # Scale
    scale_percent = 9
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)
    cv.imshow('Resized', resized)

    # Translation
    m = np.float32([[1, 0, 100], [0, 1, 50]])
    img_t = cv.warpAffine(img, m, (cols, rows))
    cv.imshow('Translation', img_t)


    # Rotation
    m2 = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
    img_r = cv.warpAffine(img, m2, (cols, rows))
    cv.imshow('Rotation', img_r)
    cv.imwrite('Rotation.jpg', img_r)

    # Flip
    horizontal_img = cv.flip(img, 0)
    vertical_img = cv.flip(img, 1)
    both_img = cv.flip(img, -1)
    cv.imshow('Horizontal', horizontal_img)
    cv.imwrite('Horizontal.jpg', horizontal_img)
    cv.imshow('Vertical', vertical_img)
    cv.imwrite('Vertical.jpg', vertical_img)
    cv.imshow('Both', both_img)
    cv.imwrite('Both.jpg', both_img)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()