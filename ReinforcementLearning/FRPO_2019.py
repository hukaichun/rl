import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

obs_dim = (18,)
act_dim = (4,)


class Controller(tf.keras.Model):
    def __init__(self, 
                 featureDims = [32,32],
                 name = "Controller"):
        super().__init__(name = name)
        self._featureMaps = [tf.keras.layers.Dense(unit, tf.nn.relu) for unit in featureDims]
        self._action_layer = tf.keras.layers.Dense(4, tf.nn.softsign)

    def call(self, state):
        feature = state
        for layer in self._featureMaps:
            feature = layer(feature)
        return self._action_layer(feature)



class Critic(tf.keras.Model):
    def __init__(self,
                 featureDims = [128,128],
                 name="critic"):
        super().__init__(name=name)
        self._featureMaps = [tf.keras.layers.Dense(unit, tf.nn.relu) for unit in featureDims]
        self._value_layer = tf.keras.layers.Dense(1)

    def call(self, state):
        feature = state
        for layer in self._featureMaps:
            feature = layer(feature)
        return self._value_layer(feature)



class Actor(tf.Module):
    def __init__(self, name="Actor"):
        self._controller = Controller()
        self._sigma = tf.Variable(0.3, name="sigma")
        self._opt = tf.keras.optimizers.Nadam(0.001)

    def pdf(self, state):
        mu = self._controller(state)
        pdf = tfp.distributions.Normal(mu, self._sigma)
        return pdf

    def action(self, state):
        pdf = self.pdf(state)
        act = pdf.sample()
        clipped_act = tf.clip_by_value(act, -1, 1)
        log_p = pdf.log_prob(clipped_act)
        marginal_log_p = tf.reduce_mean(log_p, axis=1, keepdims=True)
        return marginal_log_p, clipped_act

    def log_p(self, ref_act, state):
        pdf = self.pdf(state)
        log_p = pdf.log_prob(ref_act)
        return tf.reduce_mean(log_p, axis=1, keepdims=True)

    def loss_hinge(self, obs, ref_act, adv, log_mu, margin=tf.constant(.1,dtype=tf.float32)):
        log_pi = self.log_p(ref_act, obs)
        surr = (log_pi-log_mu)*tf.sign(adv)
        hinge = tf.minimum(surr, margin)
        return tf.reduce_mean(hinge)

    def loss_regular_l2(self):
        weight = tf.constant(0.001, dtype=tf.float32)
        tmp = [tf.reduce_sum(var**2) for var in self._controller.variables]
        vs2 = tf.reduce_sum(tmp)
        return weight*vs2

    def update(self, obs, ref_act, adv, log_mu):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            loss_main = self.loss_hinge(obs, ref_act, adv, log_mu)
            loss_regu = self.loss_regular_l2()
            loss_total = loss_main+loss_regu
        gradient = tape.gradient(loss_total, self.variables)
        clipped_g, norm = tf.clip_by_global_norm(gradient, 1.)
        gvs = zip(clipped_g, self.variables)
        self.opt.apply_gradients(gvs)
        clipped_sigma = tf.clip_by_value(self._sigma, 0.1, 0.5)
        self._sigma.assign(clipped_sigma)
        return loss_main


