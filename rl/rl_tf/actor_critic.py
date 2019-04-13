import tensorflow as tf

from .core.actor_critic_architecture import ActorCriticArchitecture
from .core import utils

class _ActorCritic(ActorCriticArchitecture):
    def __init__(self,
        obs,
        plh, 
        model_policy, 
        model_value
    ):
        super().__init__()
        self.obs = obs
        self.plh = plh
        self.model["policy"] = model_policy
        self.model["value"] = model_value

        assert "state" in plh

        with tf.name_scope("respond"):
            self.__build_respond()

        with tf.name_scope("neural_network_models"):
            self.__build_policy()
            self.__build_value()

        


    def __build_policy(self):
        tmp = []
        (
            self.policy["distribution"],
            self.policy["mu"],
            self.policy["sigma"]

        ) = self.model["policy"].feed(self.plh["state"], tmp)

        self.auxiliary["policy_feature"] = tmp


    def __build_value(self):
        tmp = []
        (
            self.value["value"]

        ) = self.model["value"].feed(self.plh["state"], tmp)

        self.auxiliary["value_feature"] = tmp


    def __build_respond(self):
        (
            self.respond["distribution"],
            self.respond["mu"],
            self.respond["sigma"],

        ) = self.model["policy"].feed(self.obs)

        act_tf = utils.bound_tf(self.respond["distribution"].sample(), -1, 1)
        log_prob = self.respond["distribution"].log_prob(act_tf)
        log_prob = tf.reduce_mean(log_prob, axis=1)

        act_prob = tf.tuple([act_tf, log_prob])
        self.respond["act_prob"] = act_prob




class ActorCritic(_ActorCritic):
    def __init__(self,
        obs,
        plh,
        model_policy,
        model_value,
        gamma,
        eps  
    ):
        super().__init__(obs, plh, model_policy, model_value)
        self.eps = eps

        with tf.name_scope("temporal_difference"):
            v_next = self.model["value"].feed(self.plh["next_state"])
            v_next = tf.stop_gradient(v_next)
            v_next = utils.bound_tf(v_next, -1./(1.-gamma), 1/(1.-gamma))
            
            with tf.name_scope("TD_target"):
                td_target = tf.add(
                        self.plh["reward"],
                        gamma*(1.-self.plh["terminal_flag"])*v_next,
                    )

            with tf.name_scope("TD"):
                td = tf.subtract(
                        td_target,
                        tf.stop_gradient(self.value["value"])
                    )

            self.value["TD"] = td


        with tf.name_scope("policy_difference"):
            with tf.name_scope("log_pi"):
                curr_log_prob = tf.reduce_mean(
                        self.policy["distribution"].log_prob(self.plh["action"]),
                        axis = 1
                    )

            with tf.name_scope("log_pi_mu_diff"):
                log_diff = curr_log_prob - self.plh["log_prob"]
                self.policy["log_diff"] = log_diff

            with tf.name_scope("clipped_ISweight"):
                tmp = tf.stop_gradient(log_diff)
                ratio = tf.exp(tmp)
                rho = utils.bound_tf(ratio, 0., 1.)

        with tf.name_scope("retrace"):
            A_trace, V_trace = utils.retrace_tf(
                    td,
                    self.value["value"],
                    rho,
                    self.plh["terminal_flag"],
                    gamma
                )
            self.value["advantage"] = A_trace
            self.value["value_target"] = V_trace

        self.__standard_loss()
        self.__regularize_loss()


    def __standard_loss(self):
        with tf.name_scope("Value_Loss"):
            self.losses["value_loss"] = tf.losses.mean_squared_error(
                    self.value["value_target"],
                    self.value["value"]
                )

        with tf.name_scope("Actor_Loss"):
            (
                self.losses["actor_loss"],
                self.auxiliary["policy_residual"]

            ) = utils.hinge_actor_tf(
                    self.value["advantage"],
                    self.policy["log_diff"],
                    self.eps
                )

    def __regularize_loss(self):
        with tf.name_scope("sigma_regularization"):
            over_high = tf.maximum(self.policy['sigma']-0.5, 0)
            over_low  = tf.minimum(self.policy['sigma']-0.1, 0)
            sigma_regulator = tf.reduce_sum(over_high - over_low)

        self.auxiliary["sigma_over_high"] = over_high
        self.auxiliary["sigma_over_low"] = over_low
        self.losses["sigma_regulator"] = sigma_regulator





 
