from deepface import DeepFace
import json
import cv2
import helper
import numpy as np

np.set_printoptions(suppress=True)


def scanEmotions():
    trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread('ExampleImages/Example1.jpg')

    # Must convert to greyscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    img_crop = []

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_crop.append(img[y:y + h, x:x + w])

    avg_array = np.zeros(7)
    emoDict = {"angry" : 0, "disgust" : 0, "fear" : 0, "happy" : 0, "sad" : 0, "surprise" : 0, "neutral": 0}
    for cropped in img_crop:
        # cv2.imshow('Cropped', cropped)
        obj = DeepFace.analyze(img_path=cropped, actions=['emotion'], enforce_detection=False)
        print(obj['dominant_emotion'])
        emoDict[obj['dominant_emotion']] += 1
        helper.addToAverages(avg_array, obj)
        # cv2.waitKey(0)

    print(avg_array / len(img_crop))

    return emoDict