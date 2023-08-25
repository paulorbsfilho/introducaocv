import cv2 as cv


def main():
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    # smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')
    webcam = cv.VideoCapture(1)

    while True:
        s, img = webcam.read()
        img = cv.flip(img, 180)

        faces = face_cascade.detectMultiScale(img, minNeighbors=20, minSize=(30, 30), maxSize=(300, 300))
        olhos = eye_cascade.detectMultiScale(img, minNeighbors=20, minSize=(10, 10), maxSize=(90, 90))
        # sorrisos = smile_cascade.detectMultiScale(img, minNeighbors=15, minSize=(15, 10), maxSize=(120, 90))

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        for (x, y, w, h) in olhos:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # for (x, y, w, h) in sorrisos:
        #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv.imshow('Your face', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
