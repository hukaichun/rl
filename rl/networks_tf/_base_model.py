#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf


class _BaseModel:
    def __init__(self, tag, suffix):
        self._name = tag + suffix
        
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def name(self):
        return self._name
    