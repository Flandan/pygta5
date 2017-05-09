# train_model.py

import numpy as np
from alexnet import alexnet
import tables

WIDTH = 80
HEIGHT = 30
LR = 1e-4
EPOCHS = 100
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

in_filename = "training_data_arma_analogue.h5"
in_file = tables.open_file(in_filename, mode = "r", title = "Training Data File")
table = in_file.root.Data.Frames
screens = table.cols.screen[:]
outputs = table.cols.output[:]

chunk = 20000
num_chunks = (len(train_data)-100)//chunk

test_screens = screens[-2000:]
test_outputs = outputs[-2000:]

model = alexnet(WIDTH, HEIGHT, LR)

for i in range(EPOCHS):
    for j in range(0, num_chunks):
        train_screens = screens[j*chunk:(j+1)*chunk]
        train_outputs = outputs[j*chunk:(j+1)*chunk]

        X = train_screens.reshape(-1,HEIGHT,WIDTH,1)/255.0
        Y = train_outputs.reshape(-1,1)

        test_x = test_screens.reshape(-1,HEIGHT,WIDTH,1)/255.0
        test_y = test_outputs.reshape(-1,1)

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=500, show_metric=False, run_id=MODEL_NAME)

        model.save(MODEL_NAME)



# tensorboard --logdir=foo:"C:\Python_Plays\pygta5\log"





