import tensorflow as tf
import numpy as np

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



def retrace_np(td, value, rho, flag, discountFactor):
    length = np.shape(td)[0]
    buffer_A = [None for _ in range(length)]
    buffer_V = [None for _ in range(length)]

    idx = length-1
    adv2 = 0
    while idx>-1:
        adv = td[idx] + discountFactor*(1-flag[idx])*adv2
        adv2 = rho[idx]*adv
        rv = value[idx]+adv2
        buffer_A[idx] = adv
        buffer_V[idx] = rv
        idx-=1
    return np.asarray(buffer_A), np.asarray(buffer_V)


if __name__ == "__main__":
    fake_td = np.random.random((5,1))
    fake_v  = np.random.random((5,1))
    fake_r  = np.random.random((5,1))
    fake_f  = np.random.randint(2,size=(5,1))
    print(retrace_np(fake_td, fake_v, fake_r, fake_f, 1.))
    print(retrace(fake_td, fake_v, fake_r, fake_f, 1.))
