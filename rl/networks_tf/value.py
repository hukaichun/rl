#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf


from ._base_model import _BaseModel



class FCValueModel(_BaseModel):
    def __init__(self, 
                 widths = [128,128],
                 tag="FC"):
        '''
            input:
                widths:      widths of nn   ; int list
        '''
        
        super().__init__(tag, suffix="_value")
        act = tf.nn.relu
        self._feature_maps = [tf.keras.layers.Dense(unit, act) for unit in widths]
        self._output_layer = tf.keras.layers.Dense(1)
    
    def feed(self,
             feature,
             featureNodes = []):
        '''
            input:
                feature:      input observation
                featureNodes: latent vectors
            
            return:
                value
        '''

        with tf.variable_scope(self.name):
            for mapping in self._feature_maps:
                feature = mapping(feature)
                featureNodes.append(feature)
                
            value = tf.reshape(self._output_layer(feature), (-1,))
            
            return value
