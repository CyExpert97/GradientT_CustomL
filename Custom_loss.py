import tensorflow as tf
import keras.backend as k

x = tf.random.uniform(minval=0, maxval=1, shape=(1000, 4), dtype=tf.float32)
y = tf.multiply(tf.reduce_sum(x, axis=-1), 5)   # y is a function of x


def custom_mse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


# Define custom loss
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return k.mean(k.square(y_pred - y_true) + k.square(layer), axis=-1)

    # Return a function
    return loss


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=[4], activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=custom_loss, optimizer='adam')

history = model.fit(x, y, epochs=10)
