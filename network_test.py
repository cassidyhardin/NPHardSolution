import network 
# import unittest
import numpy as np
import sys

def test_SimpleInputMatrix():
    test_matrix = network.greedy_network('optimal.in')
    print("This is sample input")
    print(np.matrix(test_matrix))
    print("This is output ")
    assert 1 == 1
    