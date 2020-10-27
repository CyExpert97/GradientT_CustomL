import keras
import keras.backend as k
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential


# Load mnist dataset
mnist = keras.datasets.mnist

# x_train is the list of images y_train is the labels assigned to each image

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalise values to range (0,1)
x_train, x_test = x_train/255.0, x_test/255.0

# y_train = [tf.float32(i) for i in y_train]
# y_test = [numpy.float16(i) for i in y_test]
y_train = k.cast(y_train, 'float32')
y_test = k.cast(y_test, 'float32')
print(x_train.shape)

# model=model.keras.Sequential()
model = Sequential()
# optimizer='adam'
# (28,28) represents the dimensions of image in pixels
input_layer = Flatten(input_shape=(28, 28))
model.add(input_layer)

# Activation function is relu
hidden_layer_1 = Dense(128, activation='relu')
model.add(hidden_layer_1)

# Percentage of nodes destroyed
hidden_layer_2 = Dropout(0.3)
model.add(hidden_layer_2)

# Activation function is softmax
output_layer = Dense(10, activation='softmax')
model.add(output_layer)


# Define custom loss
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return k.mean(k.square(y_pred - y_true) + k.square(layer), axis=-1)

    # Return a function
    return loss


# Building model with appropriate loss function and optimizer.
# Metrics is values you want to show i.e in this case accuracy
model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])

# Training sets for code with 5 iterations of training
model.fit(x_train, y_train, epochs=5)

# The final test set checking the models performance vs actual test data
score = model.evaluate(x_test, y_test)
print(' accuracy ', score[1])



# import tensorflow as tf
#
# x = tf.random.uniform(minval=0, maxval=1, shape=(1000, 4), dtype=tf.float32)
# y = tf.multiply(tf.reduce_sum(x, axis=-1), 5)   # y is a function of x
#
#
# def custom_mse(y_true, y_pred):
#     squared_difference = tf.square(y_true - y_pred)
#     return tf.reduce_mean(squared_difference, axis=-1)
#
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(16, input_shape=[4], activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
#
# model.compile(loss=custom_mse, optimizer='adam')
#
# history = model.fit(x, y, epochs=10)
