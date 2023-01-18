# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 21:02:04 2020

@author: Abida
"""

import numpy as np
import math
import gmpy2
from helpfunctions import *
#any number can be written as a^2+b^2+c^2 +d^2
gmpy2.get_context().precision = 128

# Try to decompose n into a sum of two squares
# Precondition: n is a probable prime, n % 4 = 1
#	(a, b, True) with a^2 + b^2 = n iff decomposition was successful
#	(0, 0, False) iff decomposition was not successful

# def	decomposeProbablePrime(n):
# 	b, r = iunit(n)
# 	if	r:
# 		return 0, b, True
# 	if	(b * b + 1) % n:		# Check whether we got an imaginary unit
# 		return 0, 0, False		# Indicate failure if not
# 	a = n
# 	while b * b > n:
# 		a, b = b, a % b
# 	return b, a % b, True


#checks if n = 4**a (8a+7)
def ThreeSqCheck(n):
    if (n%2 ==1):
        #check
        if (n%8 ==7):
            return False
        else:
            return True
    else:
        (ThreeSq(n//2))


def ThreeSq(n):
    
    # if(n%8 == 3):
    #     #try to represent n = x^2+2p
    
    # if(n%8 ==1 or n%8==2):
    
    print("threesqn =", n)
    max_a = int(math.floor(gmpy2.sqrt(n)))
    n3 = math.ceil(n// 3)
    
    
    min_a = int(math.ceil(gmpy2.sqrt(n3)))

    
    for a in range(max_a-1, min_a -1, -1):
        new_n = n - pow(a,2)
        max_b = int(math.ceil(gmpy2.sqrt(new_n)))
        min_b  = int(math.ceil(gmpy2.sqrt(new_n // 2)))
        # print("min_b:", min_b)
        # print("max_b:", max_b)              
        # if(new_n%4 ==1):
        #     x,y, val = decomposeProbablePrime(new_n)
        
        #     if val:
        #         return x,y
        
        
        #print("min_b:", min_b)
            
        # if (is_prime(new_n)):
        #     print(".")
            
        for b in range(max_b-1, min_b-1, -1):
#            print('.', end='')
            a2 = pow(a,2)
            b2 = pow(b,2) 
            d1 = n-a2-b2
            soln = gmpy2.sqrt(d1)
            #print('.', end='')
            if gmpy2.is_square(d1):
                c = int(soln)
                return a, b, c
    
                    

def FourSq(n):
    n4 = math.ceil(n//4)
    print("foursqn =", n)
    sqrtn = math.floor(gmpy2.sqrt(n))
    max_a = int(sqrtn)
    
    
    # first value of a such that a^2 >= n/4
    sqrtn4 = math.ceil(gmpy2.sqrt(n4))
    min_a = int(sqrtn4)
    
    # while pow_a < n4:
    #     min_a+=1
    #     pow_a = pow(min_a,2)
    

    #for a in range(max_a, min_a, -1):
    for a in range(max_a-1, min_a-1, -1):
        new_n = n- pow(a,2)
        
        if (ThreeSqCheck(new_n)):
            b,c,d = ThreeSq(new_n)
            print("end FourSq")
            return a,b,c,d
        
        # else:
        #     a+=1

# def	iunit(p):
# 	if	p & 7 == 5:
# 		q = 2
# 	else:
# 		primes = oddPrimes()
# 		q = next(primes)						# q is an odd (probable) prime
# 		# (q % 2 = 1) and (p % 4 = 1) implies jacobi(q, p) = jacobi(p % q, q)
# 		while jacobi(p % q, q) == 1:			# jacobi(q, p)
# 			q = primes.next()
# 			if	(q == 229) and isProbableSquare(p):# loop is running quite long
# 				# reached for p = dp(1, 4*pp(k), 1) for k > 47 or p = 1093 ** 2
# 				s, r = isqrt(p)					# check, if p is a square
# 				if	r == 0:						# if yes, return square root
# 					return s, True
# 	return pow(q, p >> 2, p), False

# def	decomposeProbablePrime(n):
# 	b, r = iunit(n)
# 	if	r:
# 		return 0, b, True
# 	if	(b * b + 1) % n:		# Check whether we got an imaginary unit
# 		return 0, 0, False		# Indicate failure if not
# 	a = n
# 	while b * b > n:
# 		a, b = b, a % b
# 	return b, a % b, True



# s = 275554457538753386855619057295806372571


# # #s = 89439920540327893572905817216531848359

# # #s = 6543654654327137
# # #s = 103370314517519252374080822183072493543
# # #s = 186461513048842853642330805668042626721
# ans  = FourSq(s)
# a,b,c,d = ans
# #ans2 = a*
# tot = a**2+b**2+c**2+d**2

# print(ans)
# print(tot)
# print(tot == s)
