import tensorflow as tf
import threading
import numpy as np
from termcolor import colored

from rl.core.actor_critic import ActorCritic
from rl.core              import utils


class FRPO(ActorCritic):
    def __init__(self,
        obs,
        plh,
        policy_model,
        value_model,
        ER,
        **kwargs
    ):
        super().__init__(obs, plh, policy_model, value_model, **kwargs)

        self.summary_period = 200
        self.ER_training = threading.Event()
        self.ER_training.clear()
        self.ER_training_coord = tf.train.Coordinator()
        self.ER = ER

        self.eps = 0.1
        self.lr = 0.001


        with tf.name_scope("IMPALA_retrace"):
            A_trace, V_trace = self.build_retrace(plh["terminal_flag"], self.gamma)
            A_trace.set_shape(plh["terminal_flag"].shape)
            V_trace.set_shape(plh["terminal_flag"].shape)

        with tf.name_scope("losses"):
            v_loss, p_loss = self.build_loss(V_trace, A_trace, self.eps)

        with tf.name_scope("TRAIN"):
            with tf.name_scope("train_policy"):
                train_p = self.build_optimizer(p_loss, self.policy_model.vars, self.lr)
            with tf.name_scope("train_value"):
                train_v = self.build_optimizer(v_loss, self.value_model.vars, self.lr)
            self.train = tf.group([train_p, train_v])



    def build_retrace(self, flag, discount):
        return utils.retrace_tf(self.td, self.value_0, self.rho, flag, discount)


    def build_loss(self,
        V_target, 
        Adv,
        eps,
    ):
        with tf.name_scope("value_loss"):
            v_err_square = tf.square(V_target - self.value_0)
            v_loss = tf.reduce_mean(v_err_square)

        with tf.name_scope("policy_regularization"):
            over_high = tf.maximum(self.sigma_pi, 0.5)
            over_low  = tf.minimum(self.sigma_pi, 0.05)
            sigma_reg = over_high - over_low

        with tf.name_scope("policy_loss"):
            margin = eps*tf.abs(Adv)
            surrage = self.log_ratio*Adv
            with tf.name_scope("hinge"):
                hinge =tf.maximum(margin - surrage, 0)
                self.hinge = hinge

            p_loss = tf.reduce_mean(hinge)+tf.reduce_mean(sigma_reg)

        return v_loss, p_loss


    def build_optimizer(self,
        loss,
        var,
        lr
    ):
        opt = tf.train.AdamOptimizer(lr)

        with tf.name_scope("gradient_clip"):
            grad = tf.gradients(loss, var)
            clipped_grad, norm = tf.clip_by_global_norm(grad, 1)
            grad_var = zip(clipped_grad, var)

        train = opt.apply_gradients(grad_var)
        return train


    def crazy_friday(self, merged, writer, sess):
        print("crazy_friday in waitting")
        self.ER_training.wait()
        update_counter = 0
        
        print("crazy_friday start")
        while not self.ER_training_coord.should_stop():
            state, action, prob, reward, next_state, flag = self.ER.sampling()
            feed_dict = {self.plh["state"]: state,
                       self.plh["action"]: action,
                       self.plh["log_mu"]: prob,
                       self.plh["reward"]: reward,
                       self.plh["next_state"]: next_state,
                       self.plh["terminal_flag"]: flag}
            if update_counter%self.summary_period==0:
                summ, _ = sess.run([merged,self.train], feed_dict=feed_dict)
                writer.add_summary(summ, update_counter)
            else:
                sess.run(self.train, feed_dict=feed_dict)

            update_counter+=1

    def log_scalar(self, tag, value):
        return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])

    def log_histogram(self, tag, values, bins=1000):
        """
            ref:
            https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514#file-tensorboard_logging-py-L41
        """
        values = np.array(values)
        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))
        bin_edges = bin_edges[1:]
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        return tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])






class ExperiencePool:
    def __init__(self, obs_dim, act_dim, capacity=2000):
        print(colored("storage formated by (state, action, prob, rewards, next state, done flags)", "red"))
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.capacity    = capacity
        self.ptrIdx      = 0
        self.full        = 0

        self.trajectory  = [None for _ in range(capacity)]

        self._state_range      = (                        0, obs_dim)
        self._action_range     = (     self._state_range[1], self._state_range[1]+act_dim)
        self._prob_range       = (    self._action_range[1], self._action_range[1]+1)
        self._reward_range     = (      self._prob_range[1], self._prob_range[1]+1)
        self._next_state_range = (    self._reward_range[1], self._reward_range[1]+obs_dim)
        self._flag_range       = (self._next_state_range[1], self._next_state_range[1]+1)

        self._free_to_take = threading.Event()

    def store(self, trajectory):
        self._free_to_take.clear()
        self.trajectory[self.ptrIdx] = np.array(trajectory)
        self.ptrIdx += 1
        if self.ptrIdx == self.capacity:
            self.full += 1
            self.ptrIdx = 0
        self._free_to_take.set()

    def sampling(self):
        idx = self.sample_idx()
        return self.take(idx)

    def sample_idx(self):
        if self.full:
            return np.random.randint(0,self.capacity)
        else:
            return np.random.randint(0,self.ptrIdx)

    def take(self, idx):
        self._free_to_take.wait()
        traj = self.trajectory[idx]
        #traj = np.vstack(traj)
        return self.traj_to_batch(traj)

    def traj_to_batch(self, traj):
        states   = traj[:,      self._state_range[0]: self._state_range[1]]
        actions  = traj[:,     self._action_range[0]: self._action_range[1]]
        probs    = traj[:,       self._prob_range[0]: self._prob_range[1]]
        rewards  = traj[:,     self._reward_range[0]: self._reward_range[1]]
        state_s  = traj[:, self._next_state_range[0]: self._next_state_range[1]]
        flags    = traj[:,       self._flag_range[0]: self._flag_range[1]]
        return states, actions, probs.reshape(-1), rewards.reshape(-1), state_s, flags.reshape(-1)
