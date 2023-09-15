import numpy as np

y = np.array([0,255,0], dtype='uint8')

x = np.array([[[0,123,231],
             [23,67,132],
             [90,112,254],
             [0,0,12]],
              [[0,120,111],
              [54,90,87],
              [150,255,0],
              [1,67,99]]])

c = np.array([[True, False, False, True],
              [False, True, False, True]])

c = np.reshape(c, (2,4,1))

z = np.where(c, y, x)

print(x.shape)
print(c.shape)
print(z)