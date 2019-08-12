import tensorflow as tf


def RL_queue(state_shape, act_shape,
              num = 1000):

    with tf.name_scope("env"):
        obs           = tf.placeholder(tf.float32, (None,)+ state_shape, name="observation")

    plh = {}
    with tf.name_scope("experience_info"):
        plh["state"]         = tf.placeholder(tf.float32, (None,)+ state_shape, name="state")
        plh["action"]        = tf.placeholder(tf.float32, (None,)+ act_shape,   name="action")
        plh["log_mu"]        = tf.placeholder(tf.float32, (None,),              name="log_mu")
        plh["reward"]        = tf.placeholder(tf.float32, (None,),              name="reward")
        plh["next_state"]    = tf.placeholder(tf.float32, (None,)+ state_shape, name="obs_next")
        plh["terminal_flag"] = tf.placeholder(tf.float32, (None,),              name="term")
    

    with tf.name_scope("info_queue"):
        queue = tf.FIFOQueue(num,
                [tf.float32,]*6,
                names = ["state", "action", "log_mu", 
                         "reward", "next_state", "terminal_flag"]
            )

        enqueue_op = queue.enqueue(plh)
        dequeue_op = queue.dequeue()

    for key in plh:
        dequeue_op[key].set_shape(plh[key].shape)

    return obs, plh, enqueue_op, dequeue_op


