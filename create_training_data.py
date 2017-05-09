# create_training_data.py

import numpy as np
from grabber import Grabber
import cv2
import time
from getkeys import key_check
import os
import tables


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output

class Frame(tables.isDescription):
    screen  = tables.UInt8Col(shape=(30, 80))
    output  = tables.UInt8Col(shape=(3,))

def main():
    h5file = tables.open_file("training_data_arma.h5", mode = "w", title = "Training Data File")
    group = h5file.create_group("/", 'Data', 'Recorded Data')
    table = h5file.create_table(group, 'Frames', Frame, "Recorded Frames")
    frame = table.row

    # 800x600 windowed mode
    grabber = Grabber(bbox=(0, 150, 800, 450))
    for i in list(range(4))[::-1]:
        print(i+1, flush=True)
        time.sleep(1)

    last_time = time.time()
    paused = True
    i = 0
    while(True):
        time.sleep(0.01)
        elapsed = time.time() - last_time
        if (elapsed > 0.03):
            keys = key_check()
            if not paused:
                print('Frame: {}\tdt: {} seconds\t'.format(j, time.time()-last_time), flush=True, end="\r")
                last_time = time.time()

                screen = grabber.grab()
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

                # resize to something a bit more acceptable for a CNN
                screen = cv2.resize(screen, (80,30))
                
                output = keys_to_output(keys)
                
                frame['screen'] = screen
                frame['output'] = output
                frame.append()
                i = i + 1

                if (i % 1000 == 0):
                    table.flush()

            if 'T' in keys:
                if paused:
                    paused = False
                    print('\nunpaused!')
                    time.sleep(0.3)
                else:
                    print('\nPausing!')
                    paused = True
                    time.sleep(0.3)
main()