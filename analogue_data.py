# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import time
import cv2
import tables

def randomize(a):
    # Generate the permutation index array.
    arr = np.array(a)
    permutation = np.random.permutation(arr.shape[0])
    print(arr.shape, flush=True)
    time.sleep(5)
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = arr[permutation]
    return shuffled_a.tolist()

class Frame(tables.IsDescription):
    screen  = tables.UInt8Col(shape=(30, 80))      # 16-character Stringtt
    output  = tables.Float32Col()
#train_data = np.load('training_data_arma_altis.npy')
in_filename = "training_data.npy"
out_filename = "training_data_analogue.npy"

train_data = np.load(in_filename)

stear_value = 0.
out_data = []

for i in range(len(train_data)-5):
    #stear_value = stear_value - data[1][0] + data[1][2]
    if (i>=5):
        sub_data = train_data[i-5:i+5]
        left = 0.
        right = 0.
        for j in range(len(sub_data)):
            left += float(sub_data[j][1][0])
            right += float(sub_data[j][1][2])
        value = (right - left) / 10.
        """
        if (i > 90):
            print("left: {}\tright: {}\tvalue: {}".format(left, right, value), flush=True)
            time.sleep(0.1)
        """
        img = train_data[i][0]
        out_data.append([img, value])
        """
        time.sleep(0.01)
        cv2.imshow('window', cv2.resize(img, (800,300)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        """
    #img = data[0]
    #stear_value /= 20.
    #stear_value = 0.
    #out_data.append([img,stear_value])
    #if choice == [1,0,0]:
    #    lefts.append([img,choice])
    #elif choice == [0,1,0]:
    #    forwards.append([img,choice])
    #elif choice == [0,0,1]:
    #    rights.append([img,choice])

"""
#min_length = min(len(lefts), len(rights), len(fwdlefts), len(fwdrights), len(stops))
print("W: {}".format(len(forwards)), flush=True)
print("A: {}".format(len(lefts)), flush=True)
print("D: {}".format(len(rights)), flush=True)
#print("MIN: {}".format(len(fwdlefts)), flush=True)
#print("MIN: {}".format(len(fwdrights)), flush=True)
#print("MIN: {}".format(len(stops)), flush=True)
forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

final_data = forwards + lefts + rights# + fwdlefts + fwdrights + stops
shuffle(final_data)
print("Final: {}".format(len(final_data)), flush=True)
#np.save('training_data_arma_tanoa2_balanced.npy', final_data)
"""
#out_data = randomize(out_data)
np.save(out_filename, out_data)
#print(type(screens))
#out_file.flush()
#out_file.close()
#in_file.close()

