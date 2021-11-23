import cv2
import numpy as np
filepath1 = 'asl_alphabet_test/asl_alphabet_test/'
filepath2 = '_test.jpg'
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath, 0)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) 

import tensorflow as tf

model = tf.keras.models.load_model("model")

n = prepare('A_test.jpg')

prediction = model.predict(n)
print(CATEGORIES[int(np.argmax(prediction[0]))])


# for letter in CATEGORIES:
#     if letter != "del":
#         n = prepare(filepath1 + letter.lower() + filepath2)
#         prediction = model.predict(n)
#         if CATEGORIES[int(np.argmax(prediction[0]))] == letter:
#             print('for ' + letter + ' model is accurate')

# Destroys all the windows created