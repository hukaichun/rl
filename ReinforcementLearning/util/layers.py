import tensorflow as tf
import tensorflow_probability as tfp

def feedforward_nn(feature, featureDims=[32,32]):
    for unit in featureDims:
        feature = tf.keras.layers.Dense(unit, tf.nn.relu)(feature)
    return feature


# class LogGaussianPDF(tf.keras.layers.Layer):
#     def __init__(self, name="log_probability"):
#         super().__init__(name=name)

#     def call(self, inputs):
#         act ,mu, sigma = inputs
#         pdf = tfp.distributions.Normal(mu, sigma)
#         log_p = pdf.log_prob(act)
#         marginal_log_p = tf.reduce_mean(log_p, axis=1)
#         return marginal_log_p


# class RandomSampleGaussian(tf.keras.layers.Layer):
#     def __init__(self, bound, name="sampling_action"):
#         super().__init__(name=name)
#         self._bound = bound

#     def call(self, inputs):
#         mu, sigma = inputs
#         pdf = tfp.distributions.Normal(mu, sigma)
#         action_sample = pdf.sample()
#         clipped_action = tf.clip_by_value(action_sample, self._bound[0], self._bound[1])
#         marginal_log_p = LogGaussianPDF()([clipped_action,mu,sigma])
#         return marginal_log_p, clipped_action