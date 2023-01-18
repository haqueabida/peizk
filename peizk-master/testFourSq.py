# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:22:34 2020

@author: Abida
"""
import secrets
import random
import math

from helpfunctions import hash_to_prime, is_prime, shamir_trick, generate_two_large_distinct_primes, mul_inv
# from FourSq import FourSq
from interface import setup, enroll
from unittest import TestCase
import unittest

from sigma import *

#RSA_KEY_SIZE = 3072 
RSA_KEY_SIZE = 256
RSA_PRIME_SIZE = int(RSA_KEY_SIZE / 2)
p, q = generate_two_large_distinct_primes(RSA_PRIME_SIZE)
n = p*q
#TODO: is this how you choose g and h? also msg, r
g = secrets.randbelow(n)
h = secrets.randbelow(n)
phin = (p-1)*(q-1)

def genFourSq():
    x= 0
    while(x%2 ==0):
        a = secrets.randbelow(pow(2, RSA_PRIME_SIZE))
        b =  secrets.randbelow(pow(2, RSA_PRIME_SIZE))
        c =  secrets.randbelow(pow(2, RSA_PRIME_SIZE))
        d =  secrets.randbelow(pow(2, RSA_PRIME_SIZE))
        
        x= pow(a,2)+pow(b,2)+pow(c,2)+pow(d,2)
    
    
    
    return a,b,c,d,x

mylen = 1000
y = [i for i in range(mylen)]
for i in range(mylen):
    a,b,c,d,x = genFourSq()
    
    
    #is_prime(x)
   

    y[i] = (is_prime(x))


print(sum(y))

# for i in range(20):
#     x = secrets.randbelow(pow(2, 256))
#     my_id = hash_to_prime(x, 128)[0] #do not need the nonce?
#     #my_id = my_id%phin
#     print(is_prime(my_id))