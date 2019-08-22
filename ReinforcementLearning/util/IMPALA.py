import tensorflow as tf

_DTYPE = tf.float32

@tf.function(input_signature=(tf.TensorSpec(shape=(None,1), dtype=_DTYPE),
                              tf.TensorSpec(shape=(None,1), dtype=_DTYPE),
                              tf.TensorSpec(shape=(None,1), dtype=_DTYPE),
                              tf.TensorSpec(shape=(None,1), dtype=_DTYPE),
                              tf.TensorSpec(shape=(), dtype=_DTYPE),
                              tf.TensorSpec(shape=(1,), dtype=_DTYPE),))
def retrace(td, value, rho, flag, discountFactor, adv2_init=[0.]):
    length = tf.shape(td)[0]
    out_A = tf.TensorArray(_DTYPE, length)
    out_V = tf.TensorArray(_DTYPE, length)
    idx = length-1
    adv2 = adv2_init
    while idx>-1:
        tmp = (1.-flag[idx])*adv2
        adv = td[idx] + discountFactor*tmp
        adv2 = rho[idx]*adv
        rv = value[idx]+adv2
        out_A = out_A.write(idx, adv)
        out_V = out_V.write(idx, rv)
        idx -= 1
    return out_A.stack(), out_V.stack()


