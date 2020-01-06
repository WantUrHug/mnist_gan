import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.utils import save_image
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib.pyplot as plt
#仍然使用pytorch定义的
#import input_data

def to_img(x):

    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

class generator(nn.Module):
	'''
	定义生成器
	'''
	def __init__(self, input_size):
		super(generator, self).__init__()
		self.dis = nn.Sequential(
			nn.Linear(input_size, 256),
			nn.ReLU(True),
			nn.Linear(256, 256),
			nn.ReLU(True),
			nn.Linear(256, 784),
			nn.Tanh())

	def forward(self, x):
		return self.dis(x)


class discriminator(nn.Module):
	'''
	定义判别器
	'''
	def __init__(self):
		super(discriminator, self).__init__()
		self.dis = nn.Sequential(
			nn.Linear(784, 256),
			nn.LeakyReLU(0.2),
			nn.Linear(256, 256),
			nn.LeakyReLU(0.2),
			nn.Linear(256, 1),
			nn.Sigmoid())

	def forward(self, x):
		return self.dis(x)


if __name__ == "__main__":

	EPOCH_NUM = 100
	BATCH_SIZE = 128
	Z_DIMENSION = 100

	#创建一个文件夹来存放训练过程的照片
	if not os.path.exists('./img'):
	    os.mkdir('./img')

	#定义dataloader来生成训练数据
	train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
	train_loader = DataLoader(dataset = train_dataset,
						batch_size = BATCH_SIZE,
						shuffle = True)
	
	D = discriminator()
	G = generator(Z_DIMENSION)
	#转化成GPU，模型也需要，和变量一样
	if torch.cuda.is_available():
		D = D.cuda()
		G = G.cuda()

	#BCE=Binary Cross Entropy，二分类交叉熵
	criterion = nn.BCELoss()
	d_optimizer = torch.optim.Adam(D.parameters(), lr = 0.0003)
	g_optimizer = torch.optim.Adam(G.parameters(), lr = 0.0003)

	history = {}
	history["real_score"] = []
	history["fake_score"] = []
	
	for epoch in range(EPOCH_NUM):
		for i, (img, _) in enumerate(train_loader):

			#将图片数据展开成一维的
			num_img = img.size(0)
			img = img.view(num_img, -1)

			real_img = Variable(img).cuda()
			#真实数据和生成器输出的标签
			real_label = Variable(torch.ones(num_img)).cuda()
			fake_label = Variable(torch.zeros(num_img)).cuda()

			#对真实数据的图片进行判别，并计算损失
			real_out = D(real_img)
			d_loss_real = criterion(real_out, real_label)
			#对真实数据判别的结果
			real_score = real_out

			#随机生成数据，使用生成器将数据转化成一个样本，再交给判别网络去判断
			z = Variable(torch.randn(num_img, Z_DIMENSION)).cuda()
			fake_img = G(z)
			fake_out = D(fake_img)
			d_loss_fake = criterion(fake_out, fake_label)
			#是一个(BATCH_SIZE, 1)的数组，里面都是评分的结果，越接近1表示是真实的数据
			fake_score = fake_out

			#先更新判别器的参数
			d_loss = d_loss_real + d_loss_fake
			d_optimizer.zero_grad()
			d_loss.backward()
			d_optimizer.step()

			#在更新了判别器之后，要来更新生成器的参数
			z = Variable(torch.randn(num_img, Z_DIMENSION)).cuda()
			fake_img = G(z)
			output = D(fake_img)
			#以下使用的是real_label，因为需要把数据伪装成真实的，希望它看起来是真的
			g_loss = criterion(output, real_label)
			#更新生成器的参数
			g_optimizer.zero_grad()
			g_loss.backward()
			g_optimizer.step()

			if (i+1) % 200 == 0:
				history["real_score"].append(real_score.data.mean())
				history["fake_score"].append(fake_score.data.mean())
				print("Epoch %d, Step %d, G loss %.4f, D loss %.4f, D real %.4f, D fake %.4f"%(epoch, i+1, g_loss, d_loss, real_score.data.mean(), fake_score.data.mean()))

		#保存一张真实数据的图片
		if epoch == 0:
			real_images = to_img(real_img.cpu().data)
			save_image(real_images, "./img/real_images.png")
		#将每个epoch的生成器结果数据都保存下来
		if (epoch + 1)%5 == 0:
			fake_images = to_img(fake_img.cpu().data)
			save_image(fake_images, "./img/fake_images-%d.png"%(epoch + 1))

	plt.plot(history["real_score"], "b", label = "real score")
	plt.plot(history["fake_score"], "r", label = "fake score")
	plt.legend()
	plt.savefig("result.png")