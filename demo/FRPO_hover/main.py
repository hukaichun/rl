import threading
import time
import tensorflow as tf
import numpy as np

from QuadcopterControl import QuadcopterControl

from rl.alg.FRPO_2018 import FRPO, ExperiencePool, RL_plh
from rl.networks_tf.dense import GaussianPolicy, Value

tick = int(time.time())
flags = tf.app.flags
flags.DEFINE_string("proj", "FRPO_No_dead_zone_{}".format(tick), "proj name")
flags.DEFINE_string("model_dir", "./RL_models/", "Model dir")
flags.DEFINE_integer("save_period", 1000, "save period")

flags.DEFINE_integer("max_traj", 100000, "max trajectory num")
flags.DEFINE_integer("traj_len", 200, "trajectory length")
flags.DEFINE_integer("buffer_size", 5000, "replay buffer size")


FLAGS = flags.FLAGS


def main(unsed_arg):
    import os
    log_dir = "./RL_logs/" + FLAGS.proj
    model_dir = FLAGS.model_dir
    model_tag = model_dir + FLAGS.proj
    if not os.path.exists(model_dir): os.makedirs(model_dir)


    env = QuadcopterControl()
    state_shape = (18,)
    action_shape = (4,)
    obs, plh = RL_plh(state_shape, action_shape)
    policy_model = GaussianPolicy(action_shape[0])
    value_model = Value()
    ER = ExperiencePool(state_shape[0], action_shape[0], FLAGS.buffer_size)
    alg = FRPO(obs=obs, 
                plh=plh, 
                policy_model=policy_model, 
                value_model=value_model,
                ER=ER)

    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(v.name, v)
    merged = tf.summary.merge_all()


    
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./logs/{}".format(tick), sess.graph)
        sess.run(init_op)
        off_policy_train = threading.Thread(target=alg.crazy_friday, args=(merged, writer, sess,))
        off_policy_train.start()

        saver = tf.train.Saver(max_to_keep=100)
        save_period = FLAGS.save_period

        R = 0
        epi = 0
        traj_len=200
        max_traj = FLAGS.max_traj
        sigmaACC = [None for _ in range(traj_len)]

        s = env.reset()
        for trajCounter in range(max_traj+1):
            traj = []
            for i in range(traj_len):
                (act, prob), sigma = sess.run([alg.respond, alg.sigma],
                                              {obs: [s]})
                sigmaACC[i] = sigma

                s_, r, done, _ = env.step(act[0])
                item = (s, act[0], prob[0], (r*50+0.5)*2, s_, done)
                traj.append(np.hstack(item))
                R += r
                if done or i == traj_len -1:
                    summ = alg.log_scalar("Score", R)
                    writer.add_summary(summ, epi)
                    s = env.reset()
                    R = 0
                    epi += 1
                else :
                    s = s_
            summSigma = alg.log_histogram("Sigma", sigmaACC)
            writer.add_summary(summSigma, trajCounter)
            alg.ER.store(traj)
            if trajCounter > 10:
                alg.ER_training.set()
            
            if trajCounter % save_period == 0:
                saver.save(sess, model_tag+str(trajCounter))
        alg.ER_training_coord.request_stop()
        alg.ER_training_coord.join([off_policy_train])


if __name__ == "__main__":
    tf.app.run()
    

