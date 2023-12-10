print("Hello, Anaconda")
print("New testing commit")



fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)



count = 0
while count < 5:
    print("Count is:" , count)
    count += 1



def print_something():
    if 3 < 5:
        print("Dumb! 3 is always less than 5")
    else :
        print("I can never reach here")
    return
print_something()



def fibonacci(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return fibonacci (n-1) + fibonacci(n-2)



class Book:

    # __init__ is a constructor for defining the class variables
    def __init__(self, title, quantity, author, price):

        self.title = title
        self.quantity = quantity
        self.author = author

        # We make the price a private variable and create another private variable 'discount'
        self.__price = price
        self.__discount = None


    # Define some other random method for this class (get the date of book publish)
    def get_publish(self):
        day = str(self.price)
        return day + '/05/2020'

    def __repr__(self):
        return f"Book: {self.title}, Quantity: {self.quantity}, Author: {self.author}"
    
book_1 = Book('The Frog', 10, 'James Twain', 20)
book_2 = Book('Moonlight stories', 5, 'Timothy sandwich', 35)

# printing the books just tells us that the variables are objects of class Book with their memory locations
print(book_1)
print(book_2)

print(book_1.title)
# print(book_1.__discount)


# Numpy

import numpy as np

range_arr = np.arange(10)
print("An array given range is \n", range_arr, " with dimensions ", range_arr.shape, "\n")

linspace_arr = np.linspace(2.0, 3.0, num=5, endpoint=False)
print("An evenly spaced array given range is \n", linspace_arr, " with dimensions ", linspace_arr.shape, "\n")



a = np.random.randint(0, 10, size = (1,4))
print("Random integer array", a, "of shape ", a.shape)


np.random.seed(0)
a = np.random.randint(0, 10, size = (1, 4))
print("Random integer array", a, "of shape ", a.shape)


uniform_rand_arr = np.random.rand(3, 2)
print("A random array from a uniform distribution is \n", uniform_rand_arr, "with dimensions ", uniform_rand_arr.shape)


mu = 3
sigma = 2.5
sample_normal_arr = mu + sigma*np.random.randn(2, 4)
print("A random array from a gaussian distribution is \n", sample_normal_arr)
print("with mu: ", mu)
print("with sigma: ", sigma)
print("with dimensions: ", sample_normal_arr)


n = np.random.rand(4, 5, 6)
print(n[0, 2, 3])
print(n[0, :, :])
print(n[0, 0:3, 0:4])
print(n[:, 3, 4])



print("a: ", n[0::2])
print("b: ", n[0::2, 1:4, 1::2]) # upperbound exclusive



n_copy = np.copy(n)
print(f"Are the arrays the same before modification: {n_copy[2, 3] == n[2, 3]}")
print(f"Before modifying values: {n_copy[2, 3]}")

n_copy[2, 3] = 0.5
print(f"After modifying values: {n_copy[2, 3]}")
print(f"Are the arrays the same after modification: {n_copy[2, 3] == n[2, 3]}")


s = np.random.rand(3, 4, 5)
print(f"Original Shape: {s.shape}")
s.size
print(s)



k = np.random.rand(5, 6)
k1 = k.flatten()
k2 = k.flatten('F')
k4 = k.reshape(-1)



y = np.random.rand(4, 5)
print(f"Original Arrays: \n {y} \n")
print(f"Shape of Original Array: {y.shape}")
y2 = np.expand_dims(y, axis = (0, 2))
print(f"Multi-axes Expanded Array: \n {y2} \n")
print(f"Shape of Expanded Array: {y2.shape}")



array1 = np.random.randint(3, size = (3, 2, 2))
array2 = np.random.randint(4, size = (3, 2, 2))
concatenated_array1 = np.concatenate((array1, array2), axis = 0)
print("Concatenated array 1 is \n", concatenated_array1, "\n\n", "and the dimensions of the concatenated array 1 are: \n", concatenated_array1.shape)
concatenated_array2 = np.concatenate((array1, array2), axis = 1)
print("Concatenated array 2 is \n", concatenated_array2, "\n\n", "and the dimensions of the concatenated array 2 are: \n", concatenated_array2.shape)
concatenated_array3 = np.concatenate((array1, array2), axis = 2)
print("Concatenated array 3 is \n", concatenated_array3, "\n\n", "and the dimensions of the concatenated array 3 are: \n", concatenated_array3.shape)



rand_arr_1 = np.random.rand(2, 3)
rand_arr_2 = np.random.rand(2, 3)
scalar = 5.0

new_arr_4 = rand_arr_1 * rand_arr_2
print("Array 1 multiplied by Array 2 is \n", new_arr_4)

max_val = np.max(rand_arr_1)
max_idx = np.argmax(rand_arr_1, axis = 0)
mean_val = np.mean(rand_arr_1)
std_val = np.std(rand_arr_1)
norm_val = np.linalg.norm(rand_arr_1)

print("any values for random array1 > random array2:")
print((rand_arr_1 > rand_arr_2).any(), "\n")



array1 = np.random.randn(3)
array2 = np.random.randn(3)

matmul_arr = np.matmul(array1, array2)
another_arr = array1@array2
print("First multiplication method: ", matmul_arr)
print("Second multiplication method: ", another_arr)



a = np.arange(60.).reshape(3, 4, 5)
b = np.arange(24.).reshape(4, 3, 2)
print('A \'s dimension ', a.shape, '\n')
print('B \'s dimension ', b.shape, '\n')

c = np.tensordot(a,b, axes=([1,0], [0,1]))
print("A tensordot B = \n", c, 'with dimension', c.shape, '\n')
d = np.zeros((5,2))
for i in range(5):
    for j in range(2):
        for k in range(3):
            for n in range(4):
                d[i,j] += a[k,n,i] * b[n,k,j]
print(c == d)



# Pytorch

import torch
x = torch.rand(5, 3)
print(x)
y = torch.cuda.is_available()
print(y)
print(np.__version__)
print(torch.__version__)



# Datasets

import pandas as pd

tmp_array = np.ones((3,3))
np.save("tmp_array.npy", tmp_array)

x = np.load("tmp_array.npy")
print(f"Loaded array is: \n {x}")

output = pd.DataFrame()
output['id'] = np.array(range(10))
output["label"] = np.array(range(10,20))
print(output)



output.to_csv("submission.csv", index = False)
output_read = pd.read_csv("submission.csv")
print(output.head())



from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, i):
        return self.xs[i], self.ys[i]
    
xs = list(range(10))
ys = list(range(10,20))
dataset = MyDataset(xs,ys)
print(dataset[0])

for x, y in dataset:
    print("X: ", x)
    print("Y: ", y)
    break



# Dataloaders(PyTorch)

from torch.utils.data import DataLoader

input = list(range(10))
target = list(range(0, 20, 2))
print('input values: ', input)
print('target values: ', target)

dataset = MyDataset(input, target)
print("The second sample is: ", dataset[2])
print("The second sample is: ", dataset.__getitem__(2))

for x, y in DataLoader(dataset):
    print(f"batch of inputs: {x}, batch of labels: {y}")

for x, y in DataLoader(dataset, batch_size = 4):
    print(f"batch of inputs: {x}, batch of labels: {y}")

for x, y in DataLoader(dataset, batch_size=4, shuffle=True):
    print(f"batch of inputs: {x}, batch of labels: {y}")

for x, y in DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True):
    print(f"batch of inputs: {x}, batch of labels: {y}")



# Data Preprocessing

import torch
import torchaudio
import matplotlib.pyplot as plt
import IPython
import librosa

import torchvision
import torchvision.transforms as transforms
from PIL import Image

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

image, label = train_dataset[100]
plt.imshow(image)
plt.show()
print('Labels:', label)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size = 32, padding = 4),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
    transforms.ToTensor(),
])

train_dataset.transform = train_transform

augmented_image, _ = train_dataset[100]
print("shape of processed image: ", augmented_image.shape)
plt.imshow(augmented_image.permute(1,2,0))
plt.show()

all_images = []

for data in train_dataset:
    image, _ = data
    all_images.append(np.array(image))

all_images_array = np.array(all_images)

Mean = np.mean(all_images_array, axis=(0, 2, 3))
Std = np.std(all_images_array, axis=(0, 2, 3))

print('Mean:', Mean)
print("Std: ", Std)

normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = Mean, std = Std),
])

train_dataset.transform = normalize_transform
test_dataset.transform = normalize_transform

normalized_image, _ = train_dataset[100]
plt.imshow(normalized_image.permute(1,2,0))
plt.show()

# Method 1
min_val, max_val = 0, 1
plt.imshow(np.clip(normalized_image.permute(1,2,0), min_val, max_val))
plt.show()

# Method 2
unnormalized_image = normalized_image.permute(1,2,0)*Std+Mean
plt.imshow(unnormalized_image)
plt.show()











    


