import os
import struct
import numpy as np
#from PIL import Image

def train_data_reader():

	train_data = "MNIST/train-images.idx3-ubyte"
	train_label = "MNIST/train-labels.idx1-ubyte"

	with open(train_data, "rb") as bytestream:

		offset = 0
		buf = bytestream.read()
		bytestream.close()
		#print("1")
		#print(buf)
	magic, numberOfLables, num_rows, num_cols = struct.unpack_from('>llll', buf, offset)
	#print(magic)
	#print(numberOfLables)
	#print(num_rows)
	#print(num_cols)
	image_size = num_rows*num_cols
	offset += struct.calcsize('>llll')
	image = np.empty((numberOfLables, num_rows, num_cols))
	for i in range(numberOfLables):
		image[i] = np.array(struct.unpack_from('>'+str(image_size)+'B', buf, offset)).reshape((num_rows, num_cols))
		offset += struct.calcsize('>'+str(image_size)+'B')
	print(image[1])

if __name__ == "__main__":

	train_data_reader()