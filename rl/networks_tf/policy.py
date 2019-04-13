#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import tensorflow_probability as tfp

from ._base_model import _BaseModel



class FCPolicyModel(_BaseModel):
    def __init__(self, 
                 action_dim,
                 widths = [64,64],
                 tag="FC"):
        '''
            input:
                action_dim:  action dimation; int
                widths:      widths of nn   ; list
        '''
        
        super().__init__(tag, suffix="_policy")
        act = tf.nn.relu
        self._feature_maps = [tf.keras.layers.Dense(unit, act) for unit in widths]
        self._feature_maps.append(tf.keras.layers.Dense(2*action_dim))
    
    
    def feed(self,
             feature, 
             featureNodes = []):
        '''
            input:
                feature:      input observation
                featureNodes: latent vectors
            
            return:
                policy, mu, sigma
        '''
        
        with tf.variable_scope(self.name):
            for mapping in self._feature_maps:
                feature = mapping(feature)
                featureNodes.append(feature)
                
            mu, sigma = tf.split(feature, 2, axis=1)
            mu = tf.sin(mu)
            sigma = tf.exp(sigma)
            
            distribution = tfp.distributions.Normal(mu, sigma)
        return distribution, mu, sigma

