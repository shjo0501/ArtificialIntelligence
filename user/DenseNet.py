import tensorflow as tf
import numpy as np

EPOCHS = 10

class DenseUnit(tf.keras.Model):
    def __init__(self, filter_out, kernel_size):
        super(DenseUnit, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        self.concat = tf.keras.layers.Concatenate()

    def call(self, x, training=False, mask=None): # (Batch, 28, 28, C_in)
        h = self.bn(x, training = training)
        h = tf.nn.relu(h)
        h = self.conv(h)                          # (Batch, 28, 28, filter_out)
        return self.concat([x, h])                # (Batch, 28, 28, C_in + filter_out)

class DenseLayer(tf.keras.Model):
    def __init__(self, num_unit, growth_rate, kernel_size):
        super(DenseLayer, self).__init__()
        self.sequence = list()
        for idx in range(num_unit):
            self.sequence.append(DenseUnit(growth_rate, kernel_size))

    def call(self, x, training=False, mask=None):
    	for unit in self.sequence:
    	    x = unit(x, training = training)
    	return x  

class TransitionLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(TransitionLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.pool = tf.keras.layers.MaxPool2D()

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)
        return self.pool(x)

class DenseNet(tf.keras.Model):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), padding='same') # 28 x 28 x 8

        self.dn1 = DenseLayer(2, 4, (3, 3))    # 28 x 28 x 16
        self.tr1 = TransitionLayer(16, (3, 3)) # 14 x 14 x 16

        self.dn2 = DenseLayer(2, 8, (3, 3))    # 14 x 14 x 32
        self.tr2 = TransitionLayer(32, (3, 3)) # 7 x 7 x 32

        self.dn3 = DenseLayer(2, 16, (3, 3))   # 7 x 7 x 64

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation = "relu")
        self.dense2 = tf.keras.layers.Dense(10,  activation = "softmax")

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)

        x = self.dn1(x, training = training)
        x = self.tr1(x)

        x = self.dn2(x, training = training)
        x = self.tr2(x)

        x = self.dn3(x, training = training)

        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

# Implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training = False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Create model
model = DenseNet()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

