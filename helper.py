from deepface import DeepFace
import json
import cv2
import helper
import numpy as np

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