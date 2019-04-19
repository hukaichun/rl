import tensorflow as tf
import tensorflow_probability as tfp

class DenseNN:
    def __init__(self,
        output_unit,
        widths, 
        name, 
        **kwargs
    ):
        self._name = name

        KERAS_DENSE = tf.keras.layers.Dense
        act = tf.nn.relu
        self.maps = [KERAS_DENSE(unit, act) for unit in widths]
        self.maps.append(KERAS_DENSE(output_unit))

    def __call__(self, feature, buffer=[]):
        raise NotImplementedError("NotImplementedError")

    @property
    def vars(self):
        total_v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        VoI = [v for v in total_v if self._name in v.name.split("/")]
        return VoI
    @property
    def name(self):
        return self._name
    


class GaussianPolicy(DenseNN):
    def __init__(self, 
        output_unit, 
        widths=[64,64],
        name="Policy"):
        super().__init__(
            2*output_unit, 
            widths, 
            name)

    def __call__(self, 
        feature, 
        featureMap=[]
    ):
        with tf.variable_scope(self.name):
            for mapping in self.maps:
                feature = mapping(feature)
                featureMap.append(feature)

            mean, sigma = tf.split(feature, 2, axis=1)
            mean = tf.sin(mean)
            sigma = tf.exp(sigma)
            distribution = tfp.distributions.Normal(mean, sigma)
        return distribution, mean, sigma



class Value(DenseNN):
    def __init__(self,
        output_unit=1,
        widths=[128,128],
        name="Value"
    ):
        super().__init__(
            output_unit,
            widths,
            name)

    def __call__(self,
        feature,
        featureMap=[]
    ):
        with tf.variable_scope(self.name):
            for mapping in self.maps:
                feature = mapping(feature)
                featureMap.append(feature)

        return feature




    
