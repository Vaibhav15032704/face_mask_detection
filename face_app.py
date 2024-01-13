import streamlit as st
import pickle
import cv2
#from tensorflow.keras import utils
#from tensorflow.keras.models import load_model
from keras.models import load_model


import tensorflow as tf
import keras


model = load_model("facemask.h5")
st.title('Face Mask Detection System')
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if st.button('Start Detection'):
    cap = cv2.VideoCapture(0)




    def detect_face_mask1(img):
        y_pred = (model.predict(img.reshape(1, 224, 224, 3)) > 0.5).astype("int32")
        return y_pred[0][0]


    def draw_label(img, text, pos, bg_color):

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
        end_x = pos[0] + text_size[0][0] + 2
        end_y = pos[0] + text_size[0][0] - 2

        cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    def detect_face(img):
        coods = haar.detectMultiScale(img)

        return coods

    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, (224, 224))

        y_pred = detect_face_mask1(img)

        coods = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for x, y, w, h in coods:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 3)

        if y_pred == 1:
            draw_label(frame, "Mask", (10, 10), (0, 225, 0))
        else:

            draw_label(frame, "No Mask", (10, 10), (0, 0, 225))

        cv2.imshow("window", frame)
        if cv2.waitKey(1) & 0XFF == ord('x'):
            break

    cv2.destroyAllWindows()