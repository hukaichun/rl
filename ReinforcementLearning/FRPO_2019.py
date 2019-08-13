import tensorflow as tf
import tensorflow_probability as tfp
from .util import IMPALA
import numpy as np

obs_dim = (18,)
act_dim = (4,)




class Actor(tf.Module):
    def __init__(self, 
                 name="Actor"):
        super().__init__(name=name)
        (self.controller,
         self.exp_rate,
         self.pdf,
         self.act) = Actor.build_everything()

        self.w = [v for v in self.controller.variables if "kernel" in v.name]

        self.opt = tf.keras.optimizers.Adam(.001)

    def regular_exploration(self,obs,
            low=tf.constant(0.1, dtype=tf.float32),
            upp=tf.constant(0.5, dtype=tf.float32)):
        exp = self.exp_rate(obs)
        too_high = tf.maximum(exp, upp)
        too_low  = tf.minimum(exp, low)
        exp_regular = tf.reduce_mean(too_high-too_low)
        return exp_regular

    def regular_kernel_l2(self):
        ws_l2 = [tf.reduce_sum(x**2) for x in self.w]
        l2    = tf.reduce_sum(ws_l2)
        return l2*tf.constant(0.001, dtype=tf.float32)

    def hinge_loss(self, obs, act, adv, log_mu, epsilon=tf.constant(1., dtype=tf.float32)):
        with tf.name_scope("RL_hinge"):
            log_p = self.pdf([obs, act])
            log_diff = log_p - log_mu
            surr = tf.sign(adv)*log_diff
            margin = epsilon
            averaged_loss = tf.reduce_mean(tf.minimum(surr, margin))
        return -averaged_loss

    def train(self, obs, act, adv, log_mu):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            hinge_loss  = self.hinge_loss(obs, act, adv, log_mu)
            exp_regular = self.regular_exploration(obs)
            l2_reg      = self.regular_kernel_l2()
            total_loss  = hinge_loss + exp_regular + l2_reg
        gradient = tape.gradient(total_loss, self.variables)
        clipped_g, norm = tf.clip_by_global_norm(gradient, 1.)
        gvs = zip(clipped_g, self.variables)
        self.opt.apply_gradients(gvs)


    @staticmethod
    def build_everything():
        obs = tf.keras.Input(obs_dim, name="observation")
        feature = tf.keras.layers.Dense(32, tf.nn.relu)(obs)
        feature = tf.keras.layers.Dense(32, tf.nn.relu)(feature)
        action  = tf.keras.layers.Dense(4, tf.nn.softsign)(feature)
        controller = tf.keras.Model(obs, action, name="Controller")

        variance = tf.keras.layers.Dense(1, tf.nn.softplus)(feature)
        exploration = tf.keras.Model(obs, variance, name="exploration_rate")

        ref_act = tf.keras.Input(act_dim, name="reference_action")
        sigma = tf.broadcast_to(variance, tf.shape(action))
        pdf = tfp.distributions.Normal(action, sigma)
        log_p = pdf.log_prob(ref_act)
        marginal_log_p = tf.reduce_mean(log_p, axis=1)
        policy_pdf = tf.keras.Model([obs, ref_act], marginal_log_p)

        random_sample  = pdf.sample()
        clipped_sample = tf.clip_by_value(random_sample, -1, 1)
        marginal_log_p = policy_pdf([obs, clipped_sample])        
        policy_act = tf.keras.Model(obs, [marginal_log_p, clipped_sample])

        return (controller,
                exploration,
                policy_pdf,
                policy_act)



class Critic(tf.Module):
    def __init__(self, gamma, name="Critic"):
        (self.value, 
         self.advantage) = Critic.build_everything(gamma)

        self.w = [v for v in self.value.variables if "kernel" in v.name]

        self.opt = tf.keras.optimizers.Adam(.001)

    def regular_kernel_l2(self):
        ws_l2 = [tf.reduce_sum(x**2) for x in self.w]
        l2 = tf.reduce_sum(ws_l2)
        return tf.constant(.001, dtype=tf.float32)*l2

    def mean_squared_error(self, obs, Vpre):
        predict_error = Vpre - self.value(obs)
        return tf.reduce_mean(predict_error**2)

    def train(self, obs, Vtar):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            error = self.mean_squared_error(obs, Vtar)
            l2_reg = self.regular_kernel_l2()
            total_loss = error+l2_reg
        gradient = tape.gradient(total_loss, self.variables)
        cliped_g, norm = tf.clip_by_global_norm(gradient)
        gvs = zip(clipped_g, self.variables)
        self.opt.apply_gradients(gvs)

    @staticmethod
    def build_everything(gamma):
        state = tf.keras.Input(obs_dim, name="state")
        feature = tf.keras.layers.Dense(128, tf.nn.relu)(state)
        feature = tf.keras.layers.Dense(128, tf.nn.relu)(feature)
        value = tf.keras.layers.Dense(1)(feature)
        state_value = tf.keras.Model(state, value, name="state_value")

        rew        = tf.keras.Input(1, name="reward")
        next_state = tf.keras.Input(obs_dim, name="next_state")
        term_flag  = tf.keras.Input(1, name="terminal_flag")
        low = np.float32(-1./(1-gamma))
        upp = np.float32(1./(1-gamma))
        next_value = state_value(next_state)
        clipped_value = tf.clip_by_value(next_value, low, upp)
        q_value = rew+tf.constant(gamma,dtype=tf.float32)*(1.-term_flag)*clipped_value
        print(q_value.shape)
        adv = q_value-state_value(state)
        print(adv.shape)
        advantage = tf.keras.Model([state, rew, term_flag, next_state], adv)

        return state_value, advantage

