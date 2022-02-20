from deepface import DeepFace
import json
import cv2
import helper
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def addToAverages(avg_array, obj):
    avg_array[0] += obj['emotion']['angry']
    avg_array[1] += obj['emotion']['disgust']
    avg_array[2] += obj['emotion']['fear']
    avg_array[3] += obj['emotion']['happy']
    avg_array[4] += obj['emotion']['sad']
    avg_array[5] += obj['emotion']['surprise']
    avg_array[6] += obj['emotion']['neutral']

def processFrame(img):
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Detect Faces 
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    img_crop = []

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        img_crop.append(img[y:y+h, x:x+w])
    avg_array = np.zeros(7)

    for cropped in img_crop:
        #cv2.imshow('Cropped', cropped)
        obj = DeepFace.analyze(img_path = cropped, actions = ['emotion'], enforce_detection=False)
        #print(obj['dominant_emotion'])
        addToAverages(avg_array, obj)
        #cv2.waitKey(0)
    return (avg_array/len(img_crop))

def updateEmotionValues(emotions, ui,_translate):
    # QtCore.QMetaObject.invokeMethod(ui.Angry,"setText", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, "ABC"))
    # QtCore.QMetaObject.invokeMethod(ui.Angry, "setText",QtCore.Qt.QueuedConnection, QtCore.Q_ARG(int, emotions[0]))
    # QtCore.QMetaObject.invokeMethod(ui.Disgust, "setText",QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, emotions[1]))
    # QtCore.QMetaObject.invokeMethod(ui.Fear, "setText",QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, emotions[2]))
    # QtCore.QMetaObject.invokeMethod(ui.Happy, "setText",QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, emotions[3]))
    # QtCore.QMetaObject.invokeMethod(ui.Sad, "setText",QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, emotions[4]))
    # QtCore.QMetaObject.invokeMethod(ui.Surprise, "setText",QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, emotions[5]))
    # QtCore.QMetaObject.invokeMethod(ui.Neutral, "setText",QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, emotions[6]))
    emotions[np.isnan(emotions)] = 0
    print(str(int(emotions[0])))
    #ui.Angry.setText(_translate("Dialog", "0"))
    ui.Angry.setText(_translate("Dialog", str(int(emotions[0])) + "%"))
    ui.Disgust.setText(_translate("Dialog", str(int(emotions[1])) + "%"))
    ui.Fear.setText(_translate("Dialog", str(int(emotions[2])) + "%"))
    ui.Happy.setText(_translate("Dialog", str(int(emotions[3])) + "%"))
    ui.Sad.setText(_translate("Dialog", str(int(emotions[4])) + "%"))
    ui.Surprise.setText(_translate("Dialog", str(int(emotions[5])) + "%"))
    ui.Neutral.setText(_translate("Dialog", str(int(emotions[6])) + "%"))