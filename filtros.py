import cv2 as cv
from matplotlib import pyplot as plt


def main():
    img = cv.imread('images_in/bola.jpg')
    blur = cv.blur(img, (23, 23))
    gaussian_blur = cv.GaussianBlur(img, (5, 5), 0)
    median = cv.medianBlur(img, 5)

    show_image_small_scale('Original', img)
    show_image_small_scale('Blur', blur)
    show_image_small_scale('Gaussian Blur', gaussian_blur)
    show_image_small_scale('Median', median)

    cv.waitKey(0)
    cv.destroyAllWindows()


def show_image_small_scale(filename, image, scale_percent=60):
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