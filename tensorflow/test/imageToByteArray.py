import numpy as np

with open("img.png", "rb") as imageFile:
  f = imageFile.read()
  b = bytearray(f)

# print('b = ', b)

# this works
for i in range(len(b)):
	if(i < 10):
		print('byte = ', b[i])

bytestream = np.fromfile('img.png', dtype=np.uint8)

for i in range(len(bytestream)):
	if(i < 10):
		print('ubyte = ', bytestream[i])
