import numpy as np
import cv2
import time
import os
import tables

file_name = 'training_data_arma_analogue.h5'
"""
if os.path.isfile(file_name):
    print('File exists, loading data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist!')
    training_data = []
"""
def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    print(a.shape[0], flush=True)
    time.sleep(5)
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

def main():
    train_data = np.load('training_data_analogue.npy')
    print(train_data.shape)
    time.sleep(3)
    #screens, outputs = randomize(screens, outputs)
    paused = True
    #length = len(training_data)
    #print(length)
    i = 18000;#len(train_data)-2000
    while(i<len(train_data)):
        #screen = training_data[i][0]
        screen = train_data[i][0]
        #keys = training_data[i][1]
        keys = train_data[i][1]
        #print(type(screen[0][0]))
        #screen = cv2.Canny(screen, threshold1=50, threshold2=300)
        cv2.imshow('window', cv2.resize(screen, (800,300)))
        print("keys: {} \t i: {}".format(keys, i), flush=True, end='\r')
        time.sleep(0.51)
        i = i + 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
main()