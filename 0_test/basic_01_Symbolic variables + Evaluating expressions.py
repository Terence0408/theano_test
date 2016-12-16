#!/usr/bin/python
# -*- coding: utf-8 -*-

'''Symbolic variables + Evaluating expressions'''

import theano
import theano.tensor as T

'''Symbolic variables'''
'''  Theano has it's own variables and functions, defined the following'''
x = T.scalar()
y = 3*(x**2) + 1

type(y)
#  theano.tensor.var.TensorVariable
y.shape
# Shape.0
print(y)
# Elemwise{add,no_inplace}.0
theano.pprint(y)
# '((TensorConstant{3} * (<TensorType(float64, scalar)> ** TensorConstant{2})) + TensorConstant{1})'
theano.printing.debugprint(y)
#Elemwise{add,no_inplace} [id A] ''
# |Elemwise{mul,no_inplace} [id B] ''
# | |TensorConstant{3} [id C]
# | |Elemwise{pow,no_inplace} [id D] ''
# |   |<TensorType(float64, scalar)> [id E]
# |   |TensorConstant{2} [id F]
# |TensorConstant{1} [id G]





'''Evaluating expressions'''
'''  Supply a dict mapping variables to values'''
y.eval({x: 2})
f = theano.function([x], y); f(2)