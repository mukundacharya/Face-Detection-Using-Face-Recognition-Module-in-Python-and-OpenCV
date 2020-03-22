import cv2
import os
import numpy as np
import face_rec as fr
import ctypes

test_img=cv2.imread('testimages/tt.jpg')

faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

#for x,y,w,h in faces_detected:
#    img=cv2.rectangle(test_img, (x,y),(x+w,y+h),(0,255,0),3)
    
#resized_img=cv2.resize(test_img,(1000,700))

#cv2.imshow("face",resized_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
faces,faceID=fr.labels_for_training_data('trainingimages')
face_recognizer=fr.train_classifier(faces,faceID)
name={0:"Ronaldo",1:"Messi"}
face_gb=[0,0,0,0]
label_gb=''
confidence_gb=200
for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence",confidence)
    print("label",label)
    if confidence<confidence_gb:
        face_gb=face
        label_gb=label
        confidence_gb=confidence
if confidence_gb>50:
    fr.draw_rect(test_img,face_gb)
    predicted_name=name[label_gb]
    fr.put_text(test_img,'unknown',face_gb[0],face_gb[1])
else:
    fr.draw_rect(test_img,face_gb)
    predicted_name=name[label_gb]
    fr.put_text(test_img,predicted_name,face_gb[0],face_gb[1])
    
#resized_img=cv2.resize(test_img,(1000,1000))
if len(faces_detected)!=0:
    cv2.imshow("face",test_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    ctypes.windll.user32.MessageBoxW(0, "Could not detect any face", "Error", 1)
    