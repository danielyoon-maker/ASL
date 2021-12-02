
import cv2
import tensorflow as tf
import numpy as np

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def prepare(filepath): # image processing function, takes in webcam footage and processes it into a form the model can understand
    IMG_SIZE = 200
    new_array = cv2.resize(filepath, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) 

model = tf.keras.models.load_model("model") # loads the model 
webcam = cv2.VideoCapture(0) #use 0 if using inbuilt webcam

# Check if the webcam is opened correctly
if not webcam.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = webcam.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #read webcame footage as greyscale
    
    prediction = model.predict([prepare(image)]) 
    prediction = (classes[int(np.argmax(prediction[0]))]) #final prediction processing --> returns an index in the order of the list in line 6
    cv2.putText(frame,prediction,(400,400), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow('Translator', frame)
    
    c = cv2.waitKey(1) # boiler plate exit on esc code
    if c == 27: # hit esc key to stop
        break

cap.release()
cv2.destroyAllWindows()