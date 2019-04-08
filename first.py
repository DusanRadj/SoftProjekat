import numpy as np
import cv2 # OpenCV biblioteka
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from my_functions import resize_region
from my_functions import get_frames_from_video
from my_functions import find_line_by_Hough
from my_functions import nadji_konturu_linije
from my_functions import nadji_jednacinu_prave
from my_functions import get_neural_network
from my_functions import da_li_je_presao
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD

def nadji_sumu(path,ann):
    
    frames, duzina_videa = get_frames_from_video(path)

    #nalazenje koordinata linije    
    start_rgb = cv2.cvtColor(frames[0].copy(), cv2.COLOR_BGR2RGB)
    line_x1 = 0
    line_x2 = 0
    line_y1 = 0
    line_y2 = 0

    pre_previous_frame = []
    previous_frame = []
    current_frame = []
    suma = 0
    lines = find_line_by_Hough(start_rgb)

    for x in range(0, len(lines)):
        line_x1,line_y1,line_x2,line_y2 = lines[0,0,:]
        print('x1 %d , x2 %d, y1 %d, y2 %d' % (line_x1,line_x2,line_y1,line_y2))
        cv2.line(start_rgb,(line_x1,line_y1),(line_x2,line_y2),(255,0,0),2)
        break

    k,n = nadji_jednacinu_prave(line_x1,line_x2,line_y1,line_y2)
    
    count = 0

    for frame in frames:
    
        count += 5
    
        if count == duzina_videa: 
            break
        
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) 
        ret, image_binary = cv2.threshold(image_grayscale, 25, 255, cv2.THRESH_BINARY) # globalni treshold, binarizacija
        
        image_binary = 255 - image_binary  # invertovanje crne i bele boje 
        
        imgContours, sveKonture, hierarchy = cv2.findContours(image_binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
        #izbaciavanje kontura koje su u okviru neke sire konture, zbog posebnih kontura u broju
        kontureBezUnutrasnjih = []
    
        for contour in sveKonture:
        
            tl_x,tl_y,width,height = cv2.boundingRect(contour)
            temp = False

            # trazim da li postoji neka koja potpuno obuhvata trenutnu iz gornjeg fora
            for con in sveKonture:
                x,y,w,h = cv2.boundingRect(con)
                if w > 200 or h > 100:  # da ne izbaci one koji su u okviru konture linije ili cele slike
                    continue
                if x < tl_x and y < tl_y and width+tl_x < w+x and height+tl_y < h+y and w < 450 and h < 500:
                    temp = True
    
            if temp is False:
                kontureBezUnutrasnjih.append(contour)
    
        #nalazenje kontura koji su brojevi
        kontureBrojeva = []
    
        for contour in kontureBezUnutrasnjih:
            tl_x,tl_y,width,height = cv2.boundingRect(contour) 
            area = cv2.contourArea(contour)

            if area > 50 and height < 100 and height > 5 and width > 8 and width < 150:
                kontureBrojeva.append(contour)
    
    
        current_frame = []
    
        #nalazenje koji brojevi su presli liniju
        for contour in kontureBrojeva: 
            tl_x,tl_y,width,height = cv2.boundingRect(contour)
        
            if tl_x > line_x1 and tl_x < line_x2:
                jeste = da_li_je_presao(k,n,tl_x,tl_y)
                if jeste:
                    temp = True
                    for pair in previous_frame:
                        razx = pair[0] - tl_x
                        razy = pair[1] - tl_y
                        if razx < 5 or razy < 5:
                            temp = False
                        
                        for pair in pre_previous_frame:
                            razx = pair[0] - tl_x
                            razy = pair[1] - tl_y
                            if razx < 5 or razy < 5:
                                temp = False
                        
                
                    if temp == False:
                        continue
                    
                    region = image_grayscale[tl_y:tl_y+height,tl_x:tl_x+width]
                    cv2.rectangle(image_rgb,(tl_x,tl_y),(tl_x+width,tl_y+height),(255,0,0),2)
                
                    slika28 = resize_region(region)
                    slika28 = slika28.astype('float32')
                    slika28 /= 255
                
                    slika28[slika28 < 0.23] = 0
                    slika30 = slika28.reshape(1,28, 28, 1)
                
                    pred = ann.predict(slika30)
                    #cv2.imwrite("%dprepoznat%d.jpg" % (count,pred.argmax()), resize_region(region))
                    #print('U frejmu %d prosao broj %d ' % (count,pred.argmax())) #zamena za onog winnera
                    suma += pred.argmax()
                    current_frame.append([tl_x,tl_y])
                
                pre_previous_frame = previous_frame
                previous_frame = current_frame
    
        #cv2.imwrite("frame%d.jpg" % count, image_rgb)     # save frame as JPEG file

    f.write(path)
    f.write('\t')
    f.write(str(suma))
    f.write('\n')
    return suma




ann = get_neural_network()

f = open("out.txt", "w")
f.write("RA 33/2015 Dusan Radjenovic\n")
f.write("file	sum\n")

rezultat = nadji_sumu('video-0.avi',ann)
rezultat = nadji_sumu('video-1.avi',ann)
rezultat = nadji_sumu('video-2.avi',ann)
rezultat = nadji_sumu('video-3.avi',ann)
rezultat = nadji_sumu('video-4.avi',ann)
rezultat = nadji_sumu('video-5.avi',ann)
rezultat = nadji_sumu('video-6.avi',ann)
rezultat = nadji_sumu('video-7.avi',ann)
rezultat = nadji_sumu('video-8.avi',ann)
rezultat = nadji_sumu('video-9.avi',ann)


f.close()




   
