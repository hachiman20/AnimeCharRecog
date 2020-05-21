from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from keras.models import load_model
import cv2
import sys
import os.path
import glob


def classname(answer):
    if answer == 0:
        prediction = "Aqua"
    elif answer == 1:
        prediction = "Asuna"
    elif answer == 2:
        prediction = "Darkness"
    elif answer == 3:
        prediction = "Genos"
    elif answer == 4:
        prediction = "Kaguya"
    elif answer == 5:
        prediction = "Kazuma"
    elif answer == 6:
        prediction = "Killua"
    elif answer == 7:
        prediction = "Levi"
    elif answer == 8:
        prediction = "Lucy"
    elif answer == 9:
        prediction = "Makise"
    elif answer == 10:
        prediction = "Megumin"
    elif answer == 11:
        prediction = "Midoriya"
    elif answer == 12:
        prediction = "Rem"
    elif answer == 13:
        prediction = "Tanjiro"
    elif answer == 14:
        prediction = "Todoroki"
    else:
        prediction = "Zero Two"
    return prediction


text = ""


def prediction(loc):
	print(loc)
	imm = image.load_img(loc, target_size = (150,150))
	img_pred = image.img_to_array(imm)
	img_pred = np.expand_dims(img_pred, axis=0)

	Data = img_pred.astype('float32')
	Data /=255

	text=""

	array = model.predict_proba(Data)
	result = array[0]

	xx = []
	for i in range(0,16,1):
		x = (result[i], i)
		xx.append(x)

	print("check4")

	xx.sort(reverse = True)
	for i in range(0,3,1):
		text = text + classname(xx[i][1]) + " : " + str(round(xx[i][0]*100,2)) + "\n"

	print(text)
	return text


model = load_model('model.h5')
model.load_weights('weights.h5')

img = input()
np.set_printoptions(precision=5,suppress=True)


from tkinter import *
from PIL import Image, ImageTk 
root = Tk()      
canvas = Canvas(root, width = 1920, height = 1080)

canvas.pack()

img_ref = []

img2 = Image.open(img)
file = ImageTk.PhotoImage(img2)

canvas.create_image(220,100, anchor=NW, image=file)
img_ref.append(file)

width = 150
height = 150
dim = (width, height)

rndfont = 40
canvas.create_text(400, 40, font=("Purisa", rndfont), text= "INPUT")
canvas.create_text(1075, 40, font=("Purisa", rndfont), text= "OUTPUT")

rndfont2 = 18

cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
imag = cv2.imread(img, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (24, 24))
i = 0
for (x, y, w, h) in faces:
    val = max(w,h)

    new_img = imag[y:y+val,x:x+val]
    new_img = cv2.resize(new_img, dim, interpolation = cv2.INTER_AREA)
    loc = img.split(".")[0]+"_"+str(x)+"_uwu"+".jpg"
    cv2.imwrite(loc,new_img)

    imm = image.load_img(loc, target_size = (150,150))
    file = ImageTk.PhotoImage(imm)
    canvas.create_image(900, 150*i + 100 + 30*i, anchor=NW, image=file)
    img_ref.append(file)
    text = prediction(loc)
    canvas.create_text(1175, 150*i + 190 + 12*i, font=("Purisa", rndfont2), text=text)
    print("\n\n",i,"\n\n")
    i = i+1

root.mainloop()