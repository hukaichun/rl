import tensorflow as tf


class ActorCritic:
    def __init__(self,
        obs,
        plh,
        policy_model,
        value_model,
        **kwargs
    ):
        self.gamma = 0.99

        self.obs = obs
        self.plh = plh
        self.policy_model = policy_model
        self.value_model = value_model


        with tf.name_scope("Actor_Critic"):
            (self.log_pi, 
             self.value_0, 
             self.value_1) = self.build_Actor_Critic(plh["state"], 
                                                     plh["action"],
                                                     plh["next_state"])

        with tf.name_scope("respond"):
            self.respond = self.build_policy_respond(obs)

        with tf.name_scope("temproal_difference"):
            self.td = self.build_temporal_difference(plh["reward"],
                                                     plh["terminal_flag"])

        with tf.name_scope("policy_difference"):
            (self.log_ratio,
             self.rho) = self.build_policy_difference(plh["log_mu"])


    def build_policy_respond(self, 
        feature
    ):
        self.policy_featureMap = []
        (self.policy,
         self.mu,
         self.sigma) = self.policy_model(feature, self.policy_featureMap)

        action_tf = tf.clip_by_value(self.policy.sample(),-1,1)
        log_prob = tf.reduce_mean(self.policy.log_prob(action_tf), axis=1)
        respond = tf.tuple([action_tf, log_prob])
        return respond


    def build_Actor_Critic(self, 
        obs0,
        action,
        obs1
    ):
        (self.pi,
         self.mu_pi,
         self.sigma_pi) = self.policy_model(obs0)

        log_pi = tf.reduce_mean(self.pi.log_prob(action),axis=1)

        self.value_featureMap = []
        value_0 = self.value_model(obs0, self.value_featureMap)
        value_1 = self.value_model(obs1)
        return log_pi, tf.reshape(value_0,(-1,)), tf.reshape(value_1,(-1,))


    def build_temporal_difference(self,
        reward,
        term
    ):
        value_1 = tf.clip_by_value(self.value_1, -1./(1-self.gamma), 1./(1-self.gamma))
        td_target = reward + self.gamma*(1-term)*value_1
        td_target = tf.stop_gradient(td_target)
        td = td_target - self.value_0
        return td


    def build_policy_difference(self,
        log_mu,
    ):
        log_ratio = self.log_pi - log_mu 
        ratio = tf.exp(log_ratio)
        rho = tf.clip_by_value(ratio,0,1)
        return log_ratio, rho


    