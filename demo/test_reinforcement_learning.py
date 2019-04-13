import sys
sys.path.append("../")



import tensorflow as tf
import time

from rl_tf.networks.policy import FCPolicyModel
from rl_tf.networks.value import FCValueModel
from rl_tf.actor_critic import ActorCritic
from rl_tf import auxiliary

state_shape = (18,)
action_shape = (4,)

# plh = auxiliary.RL_plh(state_shape, action_shape)
policy_model = FCPolicyModel(action_shape[0])
value_model = FCValueModel()



obs, plh, enq, deq = auxiliary.RL_queue(state_shape, action_shape)
config = auxiliary.defult_config

AC = ActorCritic(obs, deq, policy_model, value_model, **config)


tick = int(time.time())
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./logs/{}'.format(tick), sess.graph)  

