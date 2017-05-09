# test_model.py

import numpy as np
from grabber import Grabber
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check
from PYXInput.virtual_controller import vController

import random

WIDTH = 80
HEIGHT = 30
LR = 1e-4
EPOCHS = 100
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

t_time = 0.09

def straight():
##    if random.randrange(4) == 2:
##        ReleaseKey(W)
##    else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)
    
with tf.device("/cpu:0"):
    tflearn.config.init_graph(gpu_memory_fraction=0.3)
    model = alexnet(WIDTH, HEIGHT, LR)
    model.load(MODEL_NAME, weights_only=True)

def main():
    MyVirtual = vController()

    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    grabber = Grabber(bbox=(0, 150, 800, 450))
    paused = True
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grabber.grab()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH,HEIGHT))

            prediction = model.predict([screen.reshape(HEIGHT,WIDTH,1)/255.0])

            xPos = prediction[0][0]
            print('X-Axis: {}\tdt: {} seconds\t'.format(xPos, time.time()-last_time), flush=True, end="\r")
            last_time = time.time()
            yPos = 1.0
            #setJoy(xPos, yPos, scale)
            MyVirtual.set_value('AxisLx', xPos)
            MyVirtual.set_value('AxisLy', yPos)
            time.sleep(0.03)

        keys = key_check()
        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(0.3)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(0.3)

        if 'Y' in keys:
            MyVirtual.set_value('BtnB', 100)
            time.sleep(0.3)
            MyVirtual.set_value('BtnB', 0)
            time.sleep(0.3)
        
        if 'I' in keys:
            MyVirtual.set_value('AxisLx', 1.0)
            time.sleep(1.0)

        if 'U' in keys:
            MyVirtual = vController()

main()       










