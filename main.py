import cv2
import pytesseract


imagem = cv2.imread('images_in/placa.png')

texto_extraido = pytesseract.image_to_string(imagem, lang='por')

print(texto_extraido)