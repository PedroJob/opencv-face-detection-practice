import cv2
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)

cv2.namedWindow("gettingface")

while True:
        k = cv2.waitKey(16)

        ret, frame = cam.read()
        #colocar imagem em gray scale

        grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #this classifier "haarcascade_frontalface_default.xml" is used to detect frontal faces in frames
        #if you want to use some different classifier, enter    https://github.com/opencv/opencv/tree/master/data/haarcascades

        faces = face_classifier.detectMultiScale(grayimg, scaleFactor=1.1, minNeighbors=2, minSize=(40,40))
        #drawn rectangle
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
            cv2.putText(frame, "FACE", (x,y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("gettingface", frame)

        if k == 27:
            break
        elif k == -1:
            continue

cam.release()
