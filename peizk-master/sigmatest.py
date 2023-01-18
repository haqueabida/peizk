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



def create_list(size):
        res = []
        for i in range(size):
                x = secrets.randbelow(pow(2, 256))
                res.append(x)
        return res
    
class SigmaTest(TestCase):

    # #TODO: these tests are right now only for completeness
    
    def test_f0(self):
        msg = secrets.randbelow(pow(2, 256))
        r = secrets.randbelow(pow(2, 256))
        C = commitment(msg, r, g,h,n)       
        setpkid = 0

        ch, s1, s2 = f0(C, msg, r, g, h, n, setpkid, tms)

        is_valid = verify_f0(C, ch, s1, s2, g, h,n, setpkid, tms)
        self.assertTrue(is_valid)
    
    def test_f1(self):
        g1 =g
        h1 = h
        g2 = pow(g,2,n)
        h2 = pow(h,2,n)
        n1 = n
        n2 = n
        x = secrets.randbelow(pow(2,256))
        r1 = secrets.randbelow(pow(2,256))
        r2 = secrets.randbelow(pow(2,256))
        C1 = commitment(x, r1, g1, h1, n1)
        C2 = commitment(x, r2, g2, h2, n2)
        setpkid = 0

        ch, D, D1, D2 = f1(C1, C2, x, r1, r2, 
                                   g1, h1, g2, h2, n1, n2, 
                                   RSA_PRIME_SIZE, setpkid, tms)
        is_valid = verify_f1(ch, C1, C2, D, D1, D2, 
                             g1,h1,g2, h2, n1,n2, 
                             RSA_PRIME_SIZE, setpkid, tms)
        
        
        # self.assertEqual(V1,W1)
        # self.assertEqual(V2,W2)
        # #ch2 = H(concat(V1, V2), RSA_PRIME_SIZE)
        # self.assertEqual(ch, ch2)
        self.assertTrue(is_valid)
        

    def test_f2(self):
        
        #picking x here but we do have a chance of collision
        #TODO: fix this method?
        sigma, tau, eps, rhoa, rhoz, rhoe = [secretsGenerator.randrange(1,n) for i in range(6)]
        k= RSA_PRIME_SIZE
        x = secrets.randbelow(pow(2, 256))
        my_id = hash_to_prime(x, RSA_PRIME_SIZE)[0]
        r = secrets.randbelow(pow(2,256))
        setpkid = pow(g, r, n)
        w = secrets.randbelow(pow(2,256))
        
        #a and b are the Bezout coefficients
        a, b = bezoute_coefficients(r, my_id)
        b= -b
        ra =secrets.randbelow(pow(2,256))
        
        
        ca = commitment(a, ra, g, h, n)
        # pow_neg(g, a, n)*pow(h, ra, n)%n

            
        #TODO: figure out how to make cB
            
        B = pow_neg(g, b,n)

        
        cB = B*pow(g,w,n)%n
        
        eps =secrets.randbelow(pow(2,256))
        re = secrets.randbelow(pow(2,256))
        ce = commitment(my_id, re, cB, h, n)

        z = w * my_id
        rz = secrets.randbelow(pow(2,256))
        cz = commitment(z, rz, g, h,n)
        
        
        c, D1, D2, D3, D4, D5, D6 = f2(ce, ca, cz, cB, my_id, re, 
       a, ra, z, rz, g,h,n,k, setpkid, tms)
        
        
        is_valid = verify_f2(c,cB, ce,cz,ca, D1, D2, D3, D4, D5,
                              D6,g,h,n, RSA_PRIME_SIZE, setpkid, tms)
        
        self.assertTrue(is_valid)
        #self.assertEqual(W1, y1)
        #self.assertEqual(W2, y2)
        #looks like y3 is working
        #self.assertEqual(W3, y3)
        #looks like y4 is working
        #self.assertEqual(W4, y4)
        
        
    def test_f3(self):
        #Pick x,y,z to commit to 
        k= RSA_PRIME_SIZE
        x = secrets.randbelow(pow(2,256))
        y = secrets.randbelow(pow(2,256))
        z = x*y
        rx = secrets.randbelow(pow(2,256))
        ry = secrets.randbelow(pow(2,256))
        rz = secrets.randbelow(pow(2,256))
        C1 = commitment(x,rx, g,h,n)
        C2 = commitment(y,ry, g,h,n)
        C3 = commitment(z,rz, g,h,n)
        setpkid =0


        e, u1, u, v1, v2, v3 = f3(C3, C1, C2, 
                                            rz, x, rx, 
                                            y, ry, 
                                            g, h, n, setpkid, tms)
        
        is_valid = verify_f3(C3, C1, C2, e, u1, u, 
                             v1,v2,v3, g,h,n, setpkid, tms)
        

        self.assertTrue(is_valid)
      
        
             
    # def test_f4(self):
    #     #Pick x,y,z to commit to 
    #     k= RSA_PRIME_SIZE
    #     x = secrets.randbelow(pow(2, 256))
    #     s = hash_to_prime(x, RSA_PRIME_SIZE)[0] #do not need the nonce?

    #     #s = 1761471161

        
    #     r = secrets.randbelow(pow(2,256))
    #     C = commitment(s,r, g,h,n)
        

    #     c, c2, alpha, beta, gamma = f4(C, s, r, g, h,n)
        
    #     is_valid = verify_f4(C, c, c2,alpha, beta, gamma, g,h,n)
        

    # #     self.assertTrue(is_valid)
        
    #     def test_f4b(self):
    #         k= RSA_PRIME_SIZE
    #         x = secrets.randbelow(pow(2, 256))
    #         s = hash_to_prime(x, RSA_PRIME_SIZE)[0] #do not need the nonce?

    #     #s = 1761471161

        
    #         r = secrets.randbelow(pow(2,256))
    #         C = commitment(s,r, g,h,n)
        
    #         print(k,s,r,C)
    #         e, z1, z2 = f4b(C, s, r, g, h,n, k)
    #         print(e,z1, z2)
    #         is_valid = verify_f4b(C, g, h, n, k, e, z1, z2)
        

    #         self.assertTrue(is_valid)

    
    

if __name__ == '__main__':
    unittest.main()