#import libraries
import cv2
import numpy as np
import face_recognition
import os

#define array n path
path = "images"
images = []
classNames = []
myList = os.listdir(path)
list_dir = []

for list in myList:
    list_dir.append(list)

#get images and label
for dir in list_dir:
    image_folder_path = os.path.join(path, dir)
    for image_path in os.listdir(image_folder_path):
        img = cv2.imread(os.path.join(image_folder_path, image_path))
        images.append(img)
        classNames.append(image_folder_path.split("\\")[1])

#encode images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')


#webcam
cam = cv2.VideoCapture(0)
while True:
    success, img = cam.read()
    imageSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imageSmall = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2RGB)

    #HOG
    facesCurrFrame = face_recognition.face_locations(imageSmall) #imageFaces
    encodesCurrFrame = face_recognition.face_encodings(imageSmall, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        #compare known images with current face (encoded)
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        #use euclidean distance to measure distance between faces
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            #increase rectangle size
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(img, "{}".format(name), (x1+6, y2+25),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

