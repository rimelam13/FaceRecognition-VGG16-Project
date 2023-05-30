import cv2
import numpy as np

#load haar cascade face detection
face_classifier = cv2.CascadeClassifier(r'F:\university\AI2\traitementImage\Projet\Face-Recognition-Using-Transfer-Learning-master\Face-Recognition-Using-Transfer-Learning-master\haarcascade_frontalface_default.xml')

#load function
def face_extractor(img):
  faces = face_classifier.detectMultiScale(img, 1.3,5)

  if faces is():
    return None

  #crop all faces found
  for(x,y,w,h) in faces:
    x = x - 10
    y = y - 10
    cropped_face = img[y:y+h+50,x:x+w+50]

  return cropped_face

#initialize webcame

cap = cv2.VideoCapture(0)
count = 0
while True:
    ret,frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame),(400,400))

        file_name_path = r'F:\University\AI2\traitementImage\ProjetFRTest\images\train\hatim/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        #put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face Not Found")
        pass

    if cv2.waitKey(1)==13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")

