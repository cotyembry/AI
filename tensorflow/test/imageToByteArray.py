with open("img.png", "rb") as imageFile:
  f = imageFile.read()
  b = bytearray(f)

# print('b = ', b)

for i in range(len(b)):
	print('byte = ', b[i])
