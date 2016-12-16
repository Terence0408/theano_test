#!/usr/bin/python
# -*- coding: utf-8 -*-

'''Shared Variables'''

import theano
import theano.tensor as T
import numpy as np


'''Updates'''
'''  Store results of function evalution'''
'''  dict mapping shared variables to new values'''


count = theano.shared(0)
new_count = count + 1
updates = {count: new_count}

f = theano.function([], count, updates=updates)



print f()
print f()
print f()
count.set_value(0)
print f()
print f()
print f()
