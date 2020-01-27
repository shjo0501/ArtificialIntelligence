# Copyright is on FastCampus.
# Only allowable for study 

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# Define network architecture
class ShallowNeuralNetwork:
    def __init__(self, inputLayer, hiddenLayer, outputLayer):
        self.weight_h = np.zeros((hiddenLayer, inputLayer), dtype=np.float32)
        self.bias_h   = np.zeros((hiddenLayer,), dtype=np.float32)
        self.weight_o = np.zeros((outputLayer, hiddenLayer), dtype=np.float32)
        self.bias_o   = np.zeros((outputLayer,), dtype=np.float32)

    def __call__(self, x):
        h = sigmoid(np.matmul(self.weight_h, x) + self.bias_h)
        return softmax(np.matmul(self.weight_o, h) + self.bias_o)

dataset = np.load('ch2_dataset.npz')
inputs = dataset['inputs']
labels = dataset['labels']

# Create Model
model = ShallowNeuralNetwork(2, 128, 10)

weights = np.load('ch2_parameters.npz')
model.weight_h = weights['W_h']
model.bias_h   = weights['b_h']
model.weight_o = weights['W_o']
model.bias_o   = weights['b_o']

outputs = list()
for data, label in zip(inputs, labels):
    output = model(data)
    outputs.append(np.argmax(output))
    #print(np.argmax(output), label)
outputs = np.stack(outputs, axis=0)

plt.figure()
for idx in range(10):
    mask = labels == idx
    plt.scatter(inputs[mask, 0], inputs[mask, 1])

plt.title('GroundTruth')
plt.show()

plt.figure()
for idx in range(10):
    mask = outputs == idx
    plt.scatter(inputs[mask, 0], inputs[mask, 1])

plt.title('SNN-Result')
plt.show()
