#!/usr/bin/python
# -*- coding: utf-8 -*-

'''Automatic differention'''

import theano
import theano.tensor as T

'''Automatic differention '''
'''Gradients are free! '''
x = T.scalar()
y = T.log(x)
gradient = T.grad(y, x)
print gradient
print gradient.eval({x: 2})
print (2 * gradient)




