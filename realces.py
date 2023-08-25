import cv2 as cv
import numpy as np


def main():
    img = cv.imread('images_in/aviao.jpg', 0)
    show_image_small_scale('Original/RBG', img)
    img_inv = cv.bitwise_not(img)
    img_norm = cv.equalizeHist(img)
    img_gamma_c = gamma_adjust(img, gamma=1.5)
    c = 255 / np.log(1 + np.max(img))
    img_log = c * (np.log(img + 1))
    img_log = np.array(img_log, dtype=np.uint8)

    show_image_small_scale('Negativa', img_inv)
    show_image_small_scale('Normalizado', img_norm)
    show_image_small_scale('Correcao de gamma', img_gamma_c)
    show_image_small_scale('Realce Logaritmico', img_log)

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


def gamma_adjust(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img, table)


if __name__ == '__main__':
    main()