import cv2
import numpy as np

imagem = cv2.imread('images_in/BlobTest.jpg')

imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

params = cv2.SimpleBlobDetector_Params()
params.minDistBetweenBlobs = 5
params.blobColor = 0
params.filterByArea = False
params.minArea = 3000
params.maxArea = 4000
# params.thresholdStep = 1
# params.minThreshold = 0
# params.maxThreshold = 255
# params.minRepeatability = 2
params.filterByCircularity = False
params.minCircularity = 0.1
params.maxCircularity = 0.6
params.filterByConvexity = False
params.minConvexity = 0.4
params.maxConvexity = 1
params.filterByInertia = True
params.minInertiaRatio = 0.6
params.maxInertiaRatio = 1

detector = cv2.SimpleBlobDetector_create(params)
pontos_chave = detector.detect(imagem_cinza)

# contornos = [np.array([p.pt], dtype=np.int0) for p in pontos_chave]

contornos = []
for p in pontos_chave:
    x, y = p.pt
    raio = p.size / 2
    contorno = np.array([
        [(x - raio, y - raio)],
        [(x + raio, y - raio)],
        [(x + raio, y + raio)],
        [(x - raio, y + raio)]
    ], dtype=np.int0)
    contornos.append(contorno)

imagem_com_contornos = cv2.drawContours(imagem.copy(), contornos, -1, (0, 0, 255), 8)

cv2.imshow("Imagem com Contornos", imagem_com_contornos)
cv2.waitKey(0)
cv2.destroyAllWindows()
