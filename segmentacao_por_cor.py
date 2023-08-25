import cv2 as cv
import numpy as np


def main():
    img = cv.imread('images_in/logo_microsoft.png', 1)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Yellow
    y_lower = np.array([20, 100, 100])
    y_upper = np.array([30, 255, 255])
    y_img = img.copy()
    y_mask = cv.inRange(img_hsv, y_lower, y_upper)
    y_res = cv.bitwise_and(y_img, y_img, mask=y_mask)

    # Red
    r_lower = np.array([0, 100, 100], dtype=np.uint8)
    r_upper = np.array([10, 255, 255], dtype=np.uint8)
    r_img = img.copy()
    r_mask = cv.inRange(img_hsv, r_lower, r_upper)
    r_res = cv.bitwise_and(r_img, r_img, mask=r_mask)

    # Blue
    b_lower = np.array([90, 100, 85])
    b_upper = np.array([130, 255, 255])
    b_img = img.copy()
    b_mask = cv.inRange(img_hsv, b_lower, b_upper)
    b_res = cv.bitwise_and(b_img, b_img, mask=b_mask)

    # Green
    g_lower = np.array([35, 100, 50])
    g_upper = np.array([85, 255, 255])
    g_img = img.copy()
    g_mask = cv.inRange(img_hsv, g_lower, g_upper)
    g_res = cv.bitwise_and(g_img, g_img, mask=g_mask)

    show_image_small_scale('Original', img)
    show_image_small_scale('HSV', img_hsv)
    show_image_small_scale('Result Yellow', y_res)
    show_image_small_scale('Result Red', r_res)
    show_image_small_scale('Result Blue', b_res)
    show_image_small_scale('Result Green', g_res)

    cv.waitKey(0)
    cv.destroyAllWindows()


def show_image_small_scale(filename, image, scale_percent=25):
    try:
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv.resize(image, dim, interpolation=cv.INTER_CUBIC)
        cv.imshow(filename, resized)
    except Exception as exc:
        msg = '\nErro ao exibir imagem\n' + 'Causa: ' + str(exc)
        print(msg)


if __name__ == '__main__':
    main()