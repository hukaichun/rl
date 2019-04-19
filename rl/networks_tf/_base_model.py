#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf


class _BaseModel:
    def __init__(self, tag, suffix):
        self._name = tag + suffix
        
    @property
    def vars(self):
        total_v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        VoI = [v for v in total_v if self._name in v.name.split("/")]
        return VoI

    @property
    def name(self):
        return self._name
    