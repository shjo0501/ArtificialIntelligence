#!/usr/bin/env python
# coding: utf-8

# ## 수치 미분을 이용한 심층 신경망 학습

# In[ ]:


import time
import numpy as np


# ## 유틸리티 함수

# In[ ]:


def _t(x):
    return np.transpose(x)

def _m(A, B):
    return np.matmul(A, B)


# ## Sigmoid 구현

# In[ ]:


class Sigmoid:
    def __init__(self):
        # TODO

    def __call__(self, x):
        # TODO

    def grad(self):
        # TODO


# ## Mean Squared Error 구현

# In[ ]:


class MeanSquaredError:
    def __init__(self):
        # TODO

    def __call__(self, h, y):
        # TODO

    def grad(self):
        # TODO


# ## 뉴런 구현

# In[ ]:


class Neuron:
    def __init__(self, W, b, a_obj):
        # TODO

    def __call__(self, x):
        # TODO

    def grad(self):
        # TODO

    def grad_W(self, dh):
        # TODO

    def grad_b(self, dh):
        # TODO


# ## 심층신경망 구현

# In[ ]:


class DNN:
    def __init__(self, hidden_depth, num_neuron, input, output, activation=Sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        # Hidden Layers
        for index in range(hidden_depth):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # Output Layer
        W, b = init_var(num_neuron, output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calc_gradient(self, loss_obj):
        # TODO


# ## 경사하강 학습법

# In[ ]:


def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = loss_obj(network(x), y)  # Forward inference
    network.calc_gradient(loss_obj)  # Back-propagation
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss


# ## 동작 테스트

# In[ ]:


x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

t = time.time()
dnn = DNN(hidden_depth=5, num_neuron=32, input=10, output=2, activation=Sigmoid)
loss_obj = MeanSquaredError()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, loss_obj, alpha=0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))


# In[ ]:




