#!/usr/bin/python
# -*- coding: utf-8 -*-

'''Shared Variables'''

import theano
import theano.tensor as T
import numpy as np

'''Shared Variables '''
''' Symbolic + Storage '''
x = theano.shared(np.zeros((2, 3), dtype=theano.config.floatX))
x
# <TensorType(float64, matrix)>

'''We can get theano based variable's value. '''
values = x.get_value()
print(values.shape)
print(values)
'''We can set theano based variable's value. '''
x.set_value(values)

((x + 2) ** 2).eval()
theano.function([], (x + 2) ** 2)()