import os
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

dataset_path = "new_dataset"
model_path = "face_model.xml"
haar_cascade_path = "XmlPadrao/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(haar_cascade_path)
''
faces = []
ids = []

names = []
current_id = 0

logging.debug("Iniciando processamento de imagens para treinamento.")

def process_image(image_path, label_id):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces_rect = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        face_roi = image[y:y+h, x:x+w] 
        faces.append(face_roi)
        ids.append(label_id)
    logging.debug(f"Processada imagem: {image_path} com label ID: {label_id}")
    
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            if label not in names:
                names.append(label)
                current_id += 1
            label_id = names.index(label)
            process_image(path, label_id)

if len(faces) == 0:
    logging.error("Nenhum rosto detectado. Verifique se as imagens estão corretas e se o classificador Haar está funcionando.")
    raise Exception("Nenhum rosto detectado. Verifique se as imagens estão corretas e se o classificador Haar está funcionando.")

logging.debug(f"{len(faces)} rostos detectados. Iniciando treinamento do modelo.")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(ids))
face_recognizer.save(model_path)

np.save("names.npy", names)
logging.debug(f"Treinamento completo. {len(faces)} rostos foram detectados e treinados.")
