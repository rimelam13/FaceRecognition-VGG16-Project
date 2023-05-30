import tkinter as tk
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition App")

        self.canvas = tk.Canvas(window, width=800, height=600)
        self.canvas.pack()

        self.model = load_model(r'C:\Users\Rime\Desktop\FaceRecoModel.h5')
        self.face_cascade = cv2.CascadeClassifier(r'F:\University\AI2\traitementImage\ProjetFRTest\images\haarcascade_frontalface_default.xml')

        #self.class_names = ['anass', 'hamza', 'hatim', 'lbachir', 'rime']

        self.video_capture = cv2.VideoCapture(0)

    def start(self):
        self.process_video()

    def process_video(self):
        _, frame = self.video_capture.read()
        faces = self.face_extractor(frame)

        if len(faces) > 0:
            for face in faces:
                cv2.imshow('face', face)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        img = img.resize((800, 600), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.window.after(10, self.process_video)

    def face_extractor(self, img):
        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
        if len(faces) == 0:
            return []

        cropped_faces = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

            cropped_face = img[y:y + h, x:x + w]
            cropped_face = cv2.resize(cropped_face, (224, 224))
            im = Image.fromarray(cropped_face, 'RGB')
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            pred = self.model.predict(img_array)

            class_index = np.argmax(pred[0])
            accuracy = pred[0][class_index]

            if accuracy > 0.5:
                class_name = class_index
                text = f"{class_name}: {accuracy:.2f}"
                cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cropped_faces.append(cropped_face)

        return cropped_faces

window = tk.Tk()
app = FaceRecognitionApp(window)
app.start()
window.mainloop()