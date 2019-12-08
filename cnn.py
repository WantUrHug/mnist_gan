import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

import input_data

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.conv2 = nn.Conv2d(6 ,16, 3)

		self.fc1 = nn.Linear(400, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		relu = F.relu(self.conv1(x))
		x = F.max_pool2d(relu, (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x

	def num_flat_features(self, x):
		size = x.size()[1:] #除了batch_size之外的维度
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class MyDataset(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
	def __getitem__(self, index):
		return self.data[index], self.labels[index]
	def __len__(self):
		return len(self.labels)

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

train_data, train_label = input_data.train_data_reader()
#print("Train data shape is ", train_data.shape)
#print("Train label shape is ", train_label.shape)
#print(train_data[0])

#train_dataset = MyDataset(train_data, train_label)
X = torch.from_numpy(train_data).float()
Y = torch.from_numpy(train_label).float()
train_dataset = TensorDataset(X, Y)

train_loader = DataLoader(dataset = train_dataset,
						batch_size = 4,
						shuffle = True)
						#num_workers = 2)

for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(train_loader, 0):
		inputs, labels = data
		inputs = Variable(inputs)
		labels = Variable(labels)

		#print(inputs)
		print(inputs.size())
		print(labels.size())
		optimizer.zero_grad()

		outputs = net(inputs)
		outputs = outputs.squeeze()
		print(outputs.size())
		
		#print(outputs)
		#print(outputs.size())
		print(labels.size())
		labels.squeeze()
		loss = criterion(outputs, labels.long())
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i%2000 == 1999:
			print('[%d, %5d] loss: %.3f' %
				(epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
print("Finished training.")