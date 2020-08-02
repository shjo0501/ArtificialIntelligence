import time
import numpy as np
epsilon = 0.0001

def _t(x):
    return np.transpose(x)

def _m(x, y):
    return np.matmul(x, y)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def mean_squared_error(h, y):
	return 1 / 2 * np.mean(np.square(h - y))

class Neuron:
    def __init__(self, W, b, a):
        self.W = W
        self.b = b
        self.a = a

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def __call__(self, x):
        return self.a(_m(_t(self.W), x) + self.b)

class DNN:
    def __init__(self, hidden_depth, input_num, hidden_num, output_num, activation = sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))
        
        self.sequence = list()
        W, b = init_var(input_num, hidden_num)
        self.sequence.append(Neuron(W, b, activation))

        for _ in range(hidden_depth - 1):
            W, b = init_var(hidden_num, hidden_num)
            self.sequence.append(Neuron(W, b, activation))

        W, b = init_var(hidden_num, output_num)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calc_gradient(self, x, y, loss_func):
        def get_new_sequence(layer_index, new_neuron):
            new_seq = list()
            for i, layer in enumerate(self.sequence):
                if i == layer_index:
                    new_seq.append(new_neuron)
                else:
                    new_seq.append(layer)
            return new_seq

        def eval_sequence(x, sequence):
            for layer in sequence:
                x = layer(x)
            return x
        
        loss = loss_func(self(x), y)
        
        for id, layer in enumerate(self.sequence):
            for w_i, w in enumerate(layer.W):
                for w_j, ww in enumerate(w):
                    W = np.copy(layer.W)
                    W[w_i][w_j] = ww + epsilon

                    new_neuron = Neuron(W, layer.b, layer.a)
                    new_sequence = get_new_sequence(id, new_neuron)
                    h = eval_sequence(x, new_sequence)

                    num_grad = (loss_func(h, y) - loss) / epsilon
                    layer.dW[w_i][w_j] = num_grad
                
                for b_i, bb in enumerate(layer.b): # iterate b
                    b = np.copy(layer.b)
                    b[b_i] = bb + epsilon
                    
                    new_neuron = Neuron(layer.W, b, layer.a)
                    new_seq = get_new_sequence(id, new_neuron)
                    h = eval_sequence(x, new_seq)
                    
                    num_grad = (loss_func(h, y) - loss) / epsilon  # (f(x+eps) - f(x)) / epsilon
                    layer.db[b_i] = num_grad

        return loss

def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = network.calc_gradient(x, y, loss_obj)
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss

x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

dnn = DNN(hidden_depth=5, input_num=10, hidden_num=32, output_num=2, activation=sigmoid)

t = time.time()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, mean_squared_error, 0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed!.'.format(time.time() - t))