import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

x = tf.Variable(1.)
y = x**2

with tf.GradientTape() as g:
    g.watch(x)
    qq = x**3
print(g.gradient(qq, x))


with tf.GradientTape() as g:
    g.watch(x)
    qq = x*y
print(g.gradient(qq, x))