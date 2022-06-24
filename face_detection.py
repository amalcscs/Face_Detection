# import cv2
# import numpy as np
# import face_recognition
# import os
# from datetime import datetime
# from PIL import ImageGrab

# path = 'c:\\users\\amal\\downloads\\amal'
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)
# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)
 
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList
 
# def markAttendance(name):
#     with open('Attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')
 
# #### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# # def captureScreen(bbox=(300,300,690+300,530+300)):
# #     capScr = np.array(ImageGrab.grab(bbox))
# #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
# #     return capScr
 
# encodeListKnown = findEncodings(images)
# print('Encoding Complete')
 
# cap = cv2.VideoCapture(0)
 
# while True:
#     success, img = cap.read()
#     #img = captureScreen()
#     # imgS = cv2.resize(img,(1400, 1000), interpolation=cv2.INTER_AREA)
#     try:
#         imgS = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_AREA)
#         print(imgS.shape)
#     except:
#         break
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
#     for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
#         #print(faceDis)
#         matchIndex = np.argmin(faceDis)
 
#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             #print(name)
#             y1,x2,y2,x1 = faceLoc
#             y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#             markAttendance(name)
 
#     cv2.imshow('Webcam',img)
#     cv2.waitKey(1)


#Simple Face Detection in Picture Step 1
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# imgElon = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_RGB2BGR)
# imgTest = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)

# faceLoc = face_recognition.face_locations(imgElon)[0]

# cv2.imshow("amal",imgElon)
# cv2.imshow("amal",imgTest)
# cv2.waitKey(0)

#Simple Face Detection in Picture Step 2
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# imgElon = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_RGB2BGR)
# imgTest = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# cv2.imshow("amal",imgElon)
# cv2.imshow("amal",imgTest)
# cv2.waitKey(0)

#Simple Face Detection in Picture Step 3 focus
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# imgElon = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_RGB2BGR)
# imgTest = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# cv2.imshow("amal",imgElon)
# cv2.imshow("amal",imgTest)
# cv2.waitKey(0)

#Simple Face Detection in Picture Step 4 finding the distance between faceloc and faceloctest
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# imgElon = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_RGB2BGR)
# imgTest = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# results = face_recognition.compare_faces([encodeElon],encodeTest)
# print(results)
# cv2.imshow("amal",imgElon)
# cv2.imshow("amal",imgTest)
# cv2.waitKey(0)

#Simple Face Detection in Picture Step 5 compare images
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# imgElon = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_RGB2BGR)
# imgTest = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\Elon_Musk_2015.jpg")
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# results = face_recognition.compare_faces([encodeElon],encodeTest)
# print(results)
# cv2.imshow("amal",imgElon)
# cv2.imshow("amal",imgTest)
# cv2.waitKey(0)

#Simple Face Detection in Picture Step 6 face distance
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# imgElon = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_RGB2BGR)
# imgTest = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)
# print(results,faceDis)

# cv2.imshow("amal",imgElon)
# cv2.imshow("amal",imgTest)
# cv2.waitKey(0)

#Simple Face Detection in Picture Step 7 face distance
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# imgElon = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_RGB2BGR)
# imgTest = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)
# cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
# print(results,faceDis)

# cv2.imshow("amal",imgElon)
# cv2.imshow("amal",imgTest)
# cv2.waitKey(0)

#Simple Face Detection Attendance Step 1
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# import os 

# path = "c:\\users\\amal\\downloads\\amal"
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)

# imgElon = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_RGB2BGR)
# imgTest = face_recognition.load_image_file("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_RGB2BGR)

#Simple Face Detection Attendance Step 2
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# import os 

# path = "c:\\users\\amal\\downloads\\amal"
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

#Simple Face Detection Attendance Step 2 Encoding
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# import os 

# path = "c:\\users\\amal\\downloads\\amal"
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))

#Simple Face Detection Attendance Step 3 webcam
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# import os 

# path = "c:\\users\\amal\\downloads\\amal"
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# encodeListKnown = findEncodings(images)
# print("Encoding Complete")

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
#     # try:
#     #     imgS = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_AREA)
#     #     print(imgS.shape)
#     # except:
#     #     break
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
#     for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
#         print(faceDis)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             print(name)
#             y1,x2,y2,x1 = faceLoc
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
#     cv2.imshow("Webcam",img)
#     cv2.waitKey(1)



#cv:resize error correction
#--------------------------
# import cv2
# image = cv2.imread("c:\\users\\amal\\downloads\\amal\\amal.jpg")
# resized_image = cv2.resize(image,(300,300))
# cv2.imshow("image",resized_image)
# cv2.waitKey(0)

#Simple Face Detection Attendance Step 4 webcam display face detecting with name
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# import os 

# path = "c:\\users\\amal\\downloads\\amal"
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# encodeListKnown = findEncodings(images)
# print("Encoding Complete")

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
#     # try:
#     #     imgS = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_AREA)
#     #     print(imgS.shape)
#     # except:
#     #     break
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
#     for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
#         print(faceDis)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             print(name)
#             y1,x2,y2,x1 = faceLoc
#             y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
#     cv2.imshow("Webcam",img)
#     cv2.waitKey(1)

#Simple Face Detection Attendance Step 5 webcam attendance
#--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# import os 
# from datetime import datetime

# path = "c:\\users\\amal\\downloads\\amal"
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# def markAttendance(name):
#     with open("Attendance.csv",'r+') as f:
#         myDataList = f.readlines()
#         print(myDataList)
# markAttendance('a')

# encodeListKnown = findEncodings(images)
# print("Encoding Complete")

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
#     # try:
#     #     imgS = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_AREA)
#     #     print(imgS.shape)
#     # except:
#     #     break
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
#     for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
#         print(faceDis)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             print(name)
#             y1,x2,y2,x1 = faceLoc
#             y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
#     cv2.imshow("Webcam",img)
#     cv2.waitKey(1)

# #Simple Face Detection Attendance Step 6 attendance time and name check Attendance.csv
# #--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# import os 
# from datetime import datetime

# path = "c:\\users\\amal\\downloads\\amal"
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# def markAttendance(name):
#     with open("Attendance.csv",'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')
#         print(myDataList)

# markAttendance('amal')

# encodeListKnown = findEncodings(images)
# print("Encoding Complete")

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
#     # try:
#     #     imgS = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_AREA)
#     #     print(imgS.shape)
#     # except:
#     #     break
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
#     for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
#         print(faceDis)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             print(name)
#             y1,x2,y2,x1 = faceLoc
#             y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
#     cv2.imshow("Webcam",img)
#     cv2.waitKey(1)


# #Simple Face Detection Attendance Step 7 attendance time and name check Attendance.csv
# #--------------------------------
# import cv2
# import numpy as np
# import face_recognition
# import os 
# from datetime import datetime

# path = "c:\\users\\amal\\downloads\\amal"
# images = []
# classNames = []
# myList = os.listdir(path)
# print(myList)

# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# def markAttendance(name):
#     with open("Attendance.csv",'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')
#         print(myDataList)


# encodeListKnown = findEncodings(images)
# print("Encoding Complete")

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
#     # try:
#     #     imgS = cv2.resize(img, (1400, 1000), interpolation=cv2.INTER_AREA)
#     #     print(imgS.shape)
#     # except:
#     #     break
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
#     for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
#         print(faceDis)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             print(name)
#             y1,x2,y2,x1 = faceLoc
#             y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#             markAttendance(name)
    
#     cv2.imshow("Webcam",img)
#     cv2.waitKey(1)


import face_recognition
import imutils
import pickle
import time
import cv2
import os
 
#find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
path = "c:\\users\\amal\\downloads\\amal"
images = []
classNames = []
myList = os.listdir(path)
 
print("Streaming started")
video_capture = cv2.VideoCapture(0)
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
         encoding)
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = myList["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
 
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()