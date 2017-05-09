# train_model.py

import numpy as np
from alexnet import alexnet
WIDTH = 80
HEIGHT = 30
LR = 1e-4
EPOCHS = 100
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

train_data = np.load('training_data_analogue.npy')

chunk = 20000
num_chunks = (len(train_data)-100)//chunk

test = train_data[-2000:]

model = alexnet(WIDTH, HEIGHT, LR)

for i in range(EPOCHS):
    for j in range(0, num_chunks):
        train = train_data[j*chunk:(j+1)*chunk]

        X = np.array([i[0] for i in train]).reshape(-1,HEIGHT,WIDTH,1)/255.0
        Y = np.array([i[1] for i in train]).reshape(-1,1)

        test_x = np.array([i[0] for i in test]).reshape(-1,HEIGHT,WIDTH,1)/255.0
        test_y = np.array([i[1] for i in test]).reshape(-1,1)

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=500, show_metric=False, run_id=MODEL_NAME)

        model.save(MODEL_NAME)



# tensorboard --logdir=foo:"C:\Python_Plays\pygta5\log"





