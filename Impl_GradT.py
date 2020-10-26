import tensorflow as tf

x = tf.random.uniform(minval=0, maxval=1, shape=(1000, 4), dtype=tf.float32)
y = tf.multiply(tf.reduce_sum(x, axis=-1), 5)   # y is a function of x


def custom_mse(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=[4], activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=custom_mse, optimizer='adam')

history = model.fit(x, y, epochs=10)

