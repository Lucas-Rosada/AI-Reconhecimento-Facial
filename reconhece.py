import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

haar_cascade_path = "XmlPadrao/haarcascade_frontalface_default.xml"

model_path = "face_model.xml"
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(model_path)
names = np.load("names.npy", allow_pickle=True)

face_cascade = cv2.CascadeClassifier(haar_cascade_path)

video_capture = cv2.VideoCapture(0)

logging.debug("Iniciando a captura de vídeo.")

while True:

    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    logging.debug("Frame capturado e convertido para escala de cinza.")
    
    scale_percent = 50 
    width = int(gray_frame.shape[1] * scale_percent / 100)
    height = int(gray_frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_gray_frame = cv2.resize(gray_frame, dim, interpolation=cv2.INTER_AREA)
    logging.debug("Frame redimensionado.")

    faces_rect = face_cascade.detectMultiScale(
        resized_gray_frame,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(30, 30)
    )
    logging.debug(f"{len(faces_rect)} faces detectadas.")

    for (x, y, w, h) in faces_rect:
        x = int(x / scale_percent * 100)
        y = int(y / scale_percent * 100)
        w = int(w / scale_percent * 100)
        h = int(h / scale_percent * 100)
        
        face_roi = gray_frame[y:y+h, x:x+w]
        id, confidence = face_recognizer.predict(face_roi)
        logging.debug(f"ID: {id}, Confiança: {confidence}")

        if confidence < 100:
            name = names[id]
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Desconhecido"
            confidence_text = f"{round(100 - confidence)}%"

        logging.debug(f"Reconhecido: {name} com confiança {confidence_text}")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence_text})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.debug("Tecla 'q' pressionada, encerrando captura de vídeo.")
        break

video_capture.release()
cv2.destroyAllWindows()
logging.debug("Captura de vídeo encerrada e janelas destruídas.")
