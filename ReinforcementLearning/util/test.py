import numpy as np
import pydot
import tensorflow as tf
import layers as ul

state = tf.keras.Input(18, name="state")
action = tf.keras.Input(4, name="action")

feature_policy = ul.feedforward_nn(state)
# feature_value = ul.feedforward_nn(state)

mu = tf.keras.layers.Dense(4, tf.nn.softsign)(feature_policy)
sigma = tf.keras.layers.Dense(4, tf.math.exp)(feature_policy)

log_p = ul.LogGaussianPDF()([action, mu, sigma])
prob, act = ul.RandomSampleGaussian([-1,1])([mu, sigma])

action_probability = tf.keras.Model([state, action], log_p)
action_sampling = tf.keras.Model(state, [prob, act])

action_probability.summary()
tf.keras.utils.plot_model(action_probability, to_file="111.png", show_shapes=True)
for v in action_probability.variables:
    print(v)

action_sampling.summary()
tf.keras.utils.plot_model(action_sampling)