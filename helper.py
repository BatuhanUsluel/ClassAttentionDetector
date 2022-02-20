from deepface import DeepFace
import json
import cv2
import helper
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import collections
import matplotlib.pyplot as plt
import time
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
plotHistoricalValues = collections.deque(np.zeros((20,7)))
ax = plt.subplot(121)

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
    plotHistoricalValues.popleft()
    plotHistoricalValues.append(emotions)

    print("Historical Values:")
    print(str(plotHistoricalValues))
    anger_array = np.array([row[0] for row in plotHistoricalValues])
    disgust_array = np.array([row[1] for row in plotHistoricalValues])
    fear_array = np.array([row[2] for row in plotHistoricalValues])
    happy_array = np.array([row[3] for row in plotHistoricalValues])
    sad_array = np.array([row[4] for row in plotHistoricalValues])
    surprise_array = np.array([row[5] for row in plotHistoricalValues])
    neutral_array = np.array([row[6] for row in plotHistoricalValues])

    ax.cla()
    print("Anger array:" + str(anger_array))
    # plot cpu


    #ax.plot(disgust_array)
    #ax.text(len(disgust_array)-1, disgust_array[-1]+2, "Disgust {}%".format(int(disgust_array[-1])))

    ax.plot(anger_array)
    ax.text(len(anger_array)-1, anger_array[-1]+2, "Anger {}%".format(int(anger_array[-1])))

    #ax.plot(fear_array)
    #ax.text(len(fear_array)-1, fear_array[-1]+2, "Fear {}%".format(int(fear_array[-1])))

    ax.plot(happy_array)
    ax.text(len(happy_array)-1, happy_array[-1]+2, "Happy {}%".format(int(happy_array[-1])))

    ax.plot(sad_array)
    ax.text(len(sad_array)-1, sad_array[-1]+2, "Sad {}%".format(int(sad_array[-1])))

    #ax.plot(surprise_array)
    #ax.text(len(surprise_array)-1, surprise_array[-1]+2, "Surprise {}%".format(int(surprise_array[-1])))

    ax.plot(neutral_array)
    ax.text(len(neutral_array)-1, neutral_array[-1]+2, "Neutral {}%".format(int(neutral_array[-1])))

    ax.scatter(len(anger_array)-1, anger_array[-1])
    ax.set_ylim(0,100)
    plt.draw()
    plt.show()
    time.sleep(0.001)