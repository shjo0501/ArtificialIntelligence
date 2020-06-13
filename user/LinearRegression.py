import tensorflow as tf

W = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))

@tf.function
def linear_model(x):
    return W * x + b

@tf.function
def mse_loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

optimizer = tf.optimizers.SGD(0.01)

@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
    y_pred = linear_model(x)
    loss = mse_loss(y, y_pred)
    gradient = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradient, [W, b]))

x_data = [1, 2, 3, 4]
y_data = [3, 6, 9, 12]

for i in range(1000):
    train_step(x_data, y_data)

test_data = [100, 300, 200, 400]

print(linear_model(test_data).numpy())