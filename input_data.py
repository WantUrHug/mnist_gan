import os
import struct
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image

def train_data_reader(size = -1):

	train_data = "MNIST/train-images.idx3-ubyte"
	train_label = "MNIST/train-labels.idx1-ubyte"

	with open(train_data, "rb") as bytestream:
		buf = bytestream.read()
	with open(train_label, "rb") as bytestream:
		label_buf = bytestream.read()	

	image_offset = 0
	label_offset = 0
	magic, numberOfDatas, num_rows, num_cols = struct.unpack_from('>llll', buf, image_offset)
	magicnumber, numberOfLables = struct.unpack_from('>ll', label_buf, label_offset)
	
	image_size = num_rows*num_cols

	image_offset += struct.calcsize('>llll')
	label_offset += struct.calcsize('>ll')
	image = np.empty((numberOfDatas, 1, num_rows, num_cols))
	num_classes = 10
	label = np.empty(numberOfLables)
	
	fmt_image = '>B'
	
	for i in range(numberOfDatas):
		
		image[i] = np.array(struct.unpack_from('>'+str(image_size)+'B', buf, image_offset)).reshape((1, num_rows, num_cols))
		#image = np.array(struct.unpack_from('>'+str(image_size)+'B', buf, image_offset)).reshape((num_rows, num_cols))
		image_offset += struct.calcsize('>'+str(image_size)+'B')

		index = np.array(struct.unpack_from(fmt_image, label_buf, label_offset))[0]
		#index = np.zeros(10, dtype = np.float)
		#index[ii] = 1.0
		label[i] = index
		label_offset += struct.calcsize(fmt_image)
	
	#之前都是缩放到0-1，现在从 0-1在变化到 -1到1
	
	return (image[:size]/255.0 - 0.5)*2, label[:size]


if __name__ == "__main__":

	data, label = train_data_reader(100)
	print(data[0])
	print(label[0])
	#for i in range(30):
	#	next(g)
