import numpy as np

for x in range(3):
    new = np.load("images/"+str(x)+'.npy')
    print(new.shape)
    print(new.max())