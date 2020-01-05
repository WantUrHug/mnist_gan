import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
#仍然使用pytorch定义的
#import input_data

def generator(input):

	input = torch.from_numpy(input)
	input_size = input.size()
	print(input_size)

	output = nn.Linear(input_size, 128)(input)
	output = nn

def discriminator(input):

	return 0

if __name__ == "__main__":

	INPUT_SIZE = 100
	test_input = np.random.rand(10,100)
	print(generator(test_input))
