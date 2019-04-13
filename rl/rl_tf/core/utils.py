import tensorflow as tf


def retrace_tf(
        td, 
        value, 
        rho, 
        flag, 
        discountFactor
    ):
    def retrace(_idx, _adv2, _td, _v, _rho, _T, _outA, _outV):
        _adv  = _td[_idx] + discountFactor*(1.-_T[_idx])*_adv2
        _adv2 = _rho[_idx]*_adv
        _rv = _v[_idx] + _adv2
        return _idx-1, _adv2, _td, _v, _rho, _T, _outA, _outV

    def condition(_idx, _adv2, _td, _v, _rho, _T, _outA, _outV):
        return _idx > -1

    length = tf.shape(td)[0]
    output = tf.while_loop(condition, retrace,
                                  [length-1, 0., 
                                   td, value, 
                                   rho, flag,
                                   tf.TensorArray(tf.float32, length),
                                   tf.TensorArray(tf.float32, length)],
                                  parallel_iterations=1,
                                  back_prop=False)
    A_trace = output[6].stack(name="A_trace")
    V_trace = output[7].stack(name="V_trace")

    return A_trace, V_trace



def hinge_actor_tf(
        adv, 
        obj, 
        eps
    ):
    margin = tf.multiply(eps, tf.abs(adv), name="margin")
    surrogate = margin - adv*obj

    with tf.name_scope("hinge_loss"):
        hinge = tf.maximum(surrogate,0)

    with tf.name_scope("policy_loss"):
        loss = tf.reduce_mean(hinge)

    return loss, hinge


def bound_tf(
        obj,
        lower_bound,
        upper_bound
    ):
    with tf.name_scope("bound_by_{}_{}".format(lower_bound, upper_bound)):
        return tf.clip_by_value(obj, lower_bound, upper_bound)


