import numpy as np
import cv2 # OpenCV biblioteka
from keras.models import Sequential
import matplotlib
import matplotlib.pyplot as plt
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
import sys
import tensorflow as tf
from keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D


# Transformisati selektovani region na sliku dimenzija 28x28 da bi svi bili isti brojevi
def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST) 


#obucavanje neuronske mreze
def get_neural_network():
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    x_train /= 255
    x_test /= 255
    
    ann = Sequential()
    ann.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    ann.add(Dense(128, activation=tf.nn.relu))
    ann.add(Dropout(0.2))
    ann.add(Dense(10,activation=tf.nn.softmax))


    ann.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])#adama
    ann.fit(x=x_train,y=y_train, epochs=10)
    
    ann.evaluate(x_test, y_test)
    
    #pred = ann.predict(x_test[3000].reshape(1, 28, 28, 1))
    #aaaasss = x_test[3000]

    return ann
    

#ucitavanje videa i razdvajanje na frejmove
def get_frames_from_video(path):
    my_video = cv2.VideoCapture(path)
    
    if not my_video.isOpened():
        print('I could not open the video!')
        sys.exit()
    
    frames = []
    success,image = my_video.read()
    count = 0
    success = True
    while success:
        success,image = my_video.read()
        count += 1
        if count % 5 == 0:
            frames.append(image)
    
    return frames, count

#nalazenje koordinata linije
def find_line_by_Hough(start_rgb):
    
    start_grayscale = cv2.cvtColor(start_rgb, cv2.COLOR_RGB2GRAY)
    ret, start_binary = cv2.threshold(start_grayscale, 25, 255, cv2.THRESH_BINARY) # globalni treshold
    
    minLineLength = 200
    maxLineGap = 10

    lines = cv2.HoughLinesP(start_binary,1,np.pi/180,100,minLineLength,maxLineGap)

    return lines

def nadji_konturu_linije(sve_konture):
    
    kontureLinije = []
    
    for oneConture in sve_konture:
        x,y,w,h = cv2.boundingRect(oneConture)
        if (w > 200 and w < 400) or (h > 100 and h < 200):  # da ne izbaci one koji su u okviru konture linije
            kontureLinije.clear() #da ne ostanu iz prethodnog frejma
            kontureLinije.append(oneConture)

    return kontureLinije

def nadji_jednacinu_prave(x1,x2,y1,y2):
    
    x1 = x1 + 3 
    x2 = x2 + 3 #5
    
    y2_minus_y1 = y2-y1
    x2_minus_x1 = x2-x1
    
    k = y2_minus_y1 / x2_minus_x1

    temp = k*x1
    temp = -temp
 
    n = temp+y1
    
    
    return k,n

def da_li_je_presao(k,n,x1,y1):
    
    # y = kx + n
    # n = y-kx
    
    moje_n = y1 - k*x1
    
    greska = moje_n - n
    
    if greska < 0:
        greska = -greska
    
    if greska < 15: 
        return True
    else:
        return False
    
