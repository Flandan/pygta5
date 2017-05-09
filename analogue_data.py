# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import time
import cv2
import tables

class Frame(tables.IsDescription):
    screen  = tables.UInt8Col(shape=(30, 80))
    output  = tables.Float32Col()

in_filename = "training_data.npy"
out_filename = "training_data_analogue.npy"

in_file = tables.open_file(in_filename, mode = "r", title = "Training Data File")
table = in_file.root.Data.Frames
screens = table.cols.screen[:]
outputs = table.cols.output[:]

out_file = tables.open_file(out_filename, mode = "w", title = "Training Data File")
group = out_file.create_group("/", 'Data', 'Recorded Data')
table = out_file.create_table(group, 'Frames', Frame, "Recorded Frames")
frame = table.row

stear_value = 0.

for i in range(5,len(train_data)-5):
    #stear_value = stear_value - data[1][0] + data[1][2]
    if (i>=5):
        sub_outputs = outputs[i-5:i+5]
        left = 0.
        right = 0.
        for j in range(len(sub_outputs)):
            left += float(sub_outputs[j][0])
            right += float(sub_outputs[j][2])
        value = (right - left) / 10.

        img = screens[i]
        
        frame['screen'] = img
        frame['output'] = value

out_file.flush()
out_file.close()
in_file.close()

