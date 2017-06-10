#Author:		John Coty Embry
#Date Created:	06-09-2017
#Last Modified:	06-09-2017

import numpy as np
import struct


#1. convert the image into a byte array
with open("img.png", "rb") as imageFile:
  f = imageFile.read()
  b = bytearray(f)


# print('type = ', type(b[0])) #to see its type

#2. convert the array of ubytes into binary data (this was from stack overflow)
binaryData = ''.join(map(chr,b))

#3. get a new file ready to be written to
newFile = open('./newFileNameToGive.ubyte', 'wb') #w I assume means 'write' but I dont know what b means

#4. actually write the binary to a file
newFile.write(binaryData) #this writes out the file as binary data
