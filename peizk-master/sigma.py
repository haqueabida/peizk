# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:04:48 2020

@author: Abida
"""
from helpfunctions import *
#TODO: understand why we need secrets
import secrets
import math

#TODO: how to add the length here?
H = hash_to_length
#TODO: decide what the length is here
#TODO: also to understand how the length works
# from FourSq import FourSq
from functools import reduce
from operator import mul

secretsGenerator = secrets.SystemRandom()
def lOfBin(x):
    return len(bin(abs(x))[2:])
def commitment(msg, r, g, h, n):
    if msg < 0:
        aa = mul_inv(pow(g, abs(msg), n),n)
    else:
        aa = pow(g, abs(msg), n)

    if r < 0:
        bb = mul_inv(pow(h, abs(r),n), n)
    else :
        bb = pow(h, r, n)
    return (aa*bb)%n



def f0(C, x, r, g, h, n, setpkid, tms):
    #TODO: select r1 and r2
    r1 = secretsGenerator.randrange(1, n)
    r2 = secretsGenerator.randrange(1,n)
    R = commitment(r1, r2, g, h, n)
    ch = H(concat(C,R,  g, h, n, setpkid, tms), RSA_PRIME_SIZE)
    s1 = r1+ch*x
    s2 = r2+ch*r
    return ch, s1, s2


def verify_f0(C, ch, s1, s2, g, h,n, setpkid, tms):
    val = (commitment(s1, s2, g,h,n) * mul_inv(pow(C, ch, n),n))%n

    cval= H(concat(C, val,  g, h, n, setpkid, tms), RSA_PRIME_SIZE)
    #c = H(concat(C,R, setpkid, tms), RSA_PRIME_SIZE)

    return ch == cval


def pref1(C1, C2, x, r1, r2, g1, h1, g2, h2, n1, n2, k, setpkid, tms):
    n = max(n1, n2)
    omega = secretsGenerator.randrange(1,n)
    eta1 = secretsGenerator.randrange(1,n)
    eta2 = secretsGenerator.randrange(1,n)
    #TODO: fix. code is hanging here somewhere.
    W1 = commitment(omega, eta1, g1, h1, n1)
    W2 = commitment(omega, eta2, g2, h2, n2)
    return omega, eta1, eta2, W1, W2


def f1(C1, C2, x, r1, r2, g1, h1, g2, h2, n1, n2, k, setpkid, tms,pre1):
    #TODO: sample omega, eta1, eta1 at random
    #check: what is the correct range for these?

    # print("W1",W1)
    # print("W2",W2)
    # W1 = np.mod(pow(g1, omega, n1) * pow(h1,eta1, n1), n1)
    # W2 = np.mod(pow(g2, omega, n2) * pow(h2,eta2, n2), n2)
    omega, eta1, eta2, W1, W2 = pre1
    ch = H(concat(C1, C2, W1, W2, g1, h1, g2, h2, n1, n2, k, setpkid, tms), k)
    D = omega +ch*x
    D1 = eta1+ ch*r1
    D2 = eta2 + ch*r2
    return ch, D, D1, D2


def verify_f1(ch, C1, C2, D, D1, D2, g1,h1,g2, h2, n1,n2, k, setpkid, tms):
    #recalculate W1, W2

    W1a = commitment(D, D1, g1, h1, n1)
    Cx_inv = mul_inv(pow(C1, ch, n1), n1)
    V1 = W1a*Cx_inv%n1

    W2a = commitment(D, D2, g2, h2,n2)
    cidx_inv = mul_inv(pow(C2, ch, n2), n2)
    V2 = W2a*cidx_inv% n2
    # print("V1",V1)
    # print("V2",V2)
    ch2 = H(concat(C1, C2, V1,V2, g1, h1, g2, h2, n1, n2, k, setpkid, tms), k)
    #return ch == H(concat(V1, V2), RSA_PRIME_SIZE)
    # print(ch2)
    # print(ch)
    return  ch2==ch



def pref2(ce, ca, cz, cB, my_id, re,
       a, ra, z, rz, g,h,n,k, setpkid, tms):
       sigma = secretsGenerator.randrange(1,k)
       tau = secretsGenerator.randrange(1,k)
       eps = secretsGenerator.randrange(1,k)
       rhoa= secretsGenerator.randrange(1,k)
       rhoz= secretsGenerator.randrange(1,k)
       rhoe= secretsGenerator.randrange(1,k)
       W1a = commitment(tau, rhoe, g, h, n)
       W1 = pow_neg(setpkid, sigma, n)*W1a %n

       W2 =  commitment(sigma,rhoa, g, h, n)
       W3 = commitment(tau, rhoz, g, h,n)
       W4 = commitment(eps, rhoe, cB, h, n)
       return sigma,tau,eps,rhoa,rhoz,rhoe,W1, W2, W3, W4

def f2(ce, ca, cz, cB, my_id, re,
       a, ra, z, rz, g,h,n,k, setpkid, tms, pre2):
    #TODO: sample sigma, tau, eps, rhoa, rhoz, rhoe
    #temporary
    sigma,tau,eps,rhoa,rhoz,rhoe, W1, W2, W3, W4 = pre2
    c = H(concat(W1, W2, W3, W4, ce, ca, cz, cB, g,h,n,k, setpkid, tms), k)
    D1 = (sigma + c*a)
    D2 = (tau + c*z)
    D3 = (eps + c*my_id)
    D4 = (rhoa + c*ra)
    D5 = (rhoz + c*rz)
    D6 = (rhoe+c*re)
    return c, D1, D2, D3, D4, D5, D6


#TODO: figure out what goes as input to the verification, what is common input?
# def verify_f2(setpk, c, cB, ce,cz,ca, D1, D2, D3, D4, D5, D6, W1, W2, W3,W4, g,h,n, k):
def verify_f2(c, cB, ce,cz,ca, D1, D2, D3, D4, D5, D6,
              g,h,n, k, setpkid, tms):
    # print(D1)
    gD1 = pow_neg(g, D1, n)
    setpkD1 = pow_neg(setpkid, D1, n)
    # gD1 = pow(g, abs(D1), n)
    # print(D1)
    # setpkD1 = pow(setpk, abs(D1), n)
    # if D1<0:
    #     #gD1 = pow(g, -D1, n)
    #     gD1 = mul_inv(gD1, n)
    #     #setpkD1 = pow(setpk, -D1, n)
    #     setpkD1 = mul_inv(setpkD1, n)

    y1 = (((setpkD1*commitment(D2, D6, g, h, n))%n)*(mul_inv(pow(ce*g, c, n),n)))%n
    y1 = y1%n

    y2 = gD1*pow(h, D4, n)*mul_inv(pow(ca, c, n), n)
    y2 = y2%n

    y3 = commitment(D2, D5, g, h, n)*mul_inv(pow(cz, c, n),n)
    y3= y3%n
    y4 = commitment(D3, D6, cB, h, n)*mul_inv(pow(ce, c, n), n)
    y4 = y4%n
    cprime = H(concat(y1, y2, y3, y4,  ce, ca, cz, cB, g,h,n,k, setpkid, tms), k)


    # print (y1 == W1, y2 == W2, y3 == W3, y4 == W4)
    return c == cprime

def pref3(C3, C1, C2, rz, x, rx, y, ry, g, h, n, setpkid, tms):
       t1 = secretsGenerator.randrange(pow(2, RSA_PRIME_SIZE))
       t2 = secretsGenerator.randrange(pow(2, RSA_PRIME_SIZE))
       s1 = secretsGenerator.randrange(pow(2, RSA_PRIME_SIZE))
       s2 = secretsGenerator.randrange(pow(2, RSA_PRIME_SIZE))
       s3 = secretsGenerator.randrange(pow(2, RSA_PRIME_SIZE))
       d1 = commitment(t1, s1, g, h, n)
       d2 = commitment(t2, s2, g,h,n)
       d3 = commitment(t2, s3, C1,h,n)

       return t1,t2,s1,s2,s3,d1, d2, d3


def f3(C3, C1, C2, rz, x, rx, y, ry, g, h, n, setpkid, tms,pre3):
    t1,t2,s1,s2,s3, d1, d2, d3 = pre3
    e = H(concat(d1,d2, d3, g, h, n, setpkid, tms), RSA_PRIME_SIZE)
    u1 = t1+e*x
    u = t2 + e*y
    v1 = s1+e*rx
    v2 = s2+e*ry
    #TODO: check if it's okay to mod by n, because v3 can end up negative so how do we deal with that?
    v3 = s3 +e*(rz-y*rx)
    return e, u1, u, v1, v2, v3


#values are not cancelling out the way I expect them to
def verify_f3(C3, C1, C2, e, u1, u, v1,v2,v3, g,h,n, setpkid, tms):

   # e = H(concat(d1,d2, d3), RSA_PRIME_SIZE) #first verifier re-calculates the challenge

    cc1a = commitment(u1, v1, g,h,n) #the whole thing
    C1_inv = mul_inv(pow(C1, e, n), n)
    #TODO: this function is hanging.
    cc1 = cc1a*C1_inv%n

    cc2 = commitment(u,v2, g,h,n)
    C2_inv = mul_inv(pow(C2, e,n),n)
    cc2 = cc2*C2_inv%n


    if v3<0:
        aa = mul_inv(pow(h, abs(v3), n), n)
    else:
        aa = pow(h, v3, n)

    C3_inv = mul_inv(pow(C3,e,n),n)

    cc3 = (pow(C1, u, n)*aa*C3_inv)%n


    #return (d1==cc1) and (d2==cc2) and (d3==cc3)
    return e == H(concat(cc1, cc2, cc3, g, h, n, setpkid, tms), RSA_PRIME_SIZE)
    #return cc1 and cc2 and cc3


def f4(C, s, r, g, h,n, k):
    #TODO:smarter way to make the list?
    r1 =[0 for i in range(4)]
    r2 = [0 for i in range(4)]
    for i in range(4):
            r2[i] = secretsGenerator.randrange(1, n)
            r1[i] = secretsGenerator.randrange(1, n)
    r2[3] = r - (r2[0]+r2[1]+r2[2])
    a,b,c,d = FourSq(s)
    si = [a,b,c,d]
    #square every element in s
    s2 = [pow(i,2) for i in si]
    #s2 = np.power(si, 2)
    c2 = [commitment(s2[i],r2[i], g, h, n) for i in range(4)]

    # print("g = ", h)
    # print("h = ", h)
    # print("r1[0]= ", r1[0])
    # print("r1[1] = ", r1[1])
    # print("r1[2] = ", r1[2])
    # print("r1[3]= ", r1[3])
    # print("si[0]= ", si[0])
    # print("si[1] = ", si[1])
    # print("si[2] = ", si[2])
    # print("si[3]= ", si[3])
    # print("r2[0]= ", r2[0])
    # print("r2[1] = ", r2[1])
    # print("r2[2] = ", r2[2])
    # print("r2[3]= ", r2[3])


    #print("h: ", h)
    c = [(commitment(si[i], r2[i], g, h, n)*mul_inv(pow_neg(h, r1[i], n), n))%n for i in range(4)]
    alpha = f0(C, s,r, g,h,n)
    beta = [f0(c[i], si[i], r2[i]-r1[i], g,h,n) for i in range(4)]
    gamma = [f3(c2[i], c[i], c[i], r2[i], si[i], r2[i]-r1[i],
                          si[i], r2[i]-r1[i], g,h,n)
                      for i in range(4)]

    return c, c2, alpha, beta, gamma


#c, c2 are vectors. beta is vector.
def verify_f4(C, c, c2,alpha, beta, gamma, g,h,n):
    R, s1, s2 = alpha #because it is output of f0
    t1 = verify_f0(C, R, s1, s2, g, h,n)

    t2 = True
    for ci, b in zip(c, beta):
        R, s1, s2 = b
        tb = verify_f0(ci, R, s1, s2, g, h,n) #each beta also comes from f0
        t2 = t2 and tb

    t3 = True

    for ci, c2i, gam in zip(c, c2, gamma):
        e, u1, u, v1, v2, v3 = gam #gamma is output of f3
        t3 =t3 and verify_f3(c2i, ci, ci, e, u1, u, v1,v2,v3, g,h,n)

    c2prod = reduce(mul, c2, 1)

    final_check = (C==(c2prod%n))

    return t1 and t2 and t3 and final_check


def f4b(C, s, r, g, h,n, k):
    #l = k/2 and to prove that s \in Z_2^l
    l = k//2
    # print(l)
    m = 20
    t = 128 #hardcoded for now
    a = secretsGenerator.randrange(1, pow(2, l+t + m))
    alpha = secretsGenerator.randrange(1,n)
    A = commitment(a, alpha, g, h, n)
    # print("here")
    e = hash_to_length(A, t)
    # print("e= ", lOfBin(e))
    # print("a= ", lOfBin(a))
    # print("s= ", lOfBin(s))

    # print("m+t+l= ", m + t+ l)
    z1 = a + s*e
    # print("z1 = ", lOfBin(z1))
    z2 = alpha + e*r

    return (e, z1, z2)

def verify_f4b(C, g, h, n, k, e, z1, z2):
    l = k//2
    m = 20
    t = 128 #hardcoded for now
    t1 = False
    t2 = False
    # print(z1 >= pow(2,t)*pow(2,l))
    # print(z1 < pow(2,t+m)*pow(2,l))
    if z1 >= pow(2,t)*pow(2,l) and z1 < pow(2,t+m)*pow(2,l):
        t1 = True
    if e == hash_to_length(commitment(z1,z2,g,h,n)*mul_inv(pow(C,e,n),n)%n, t):
        t2 = True
    # print("t2 ===== ", t2)
    return t1 and t2




# def zkVerify(stmt, proof, g, h, g1, h1):
#     setpk, C = stmt
#     cid, ca, cB, cw, cz, ce,
#     pok1, pok2, pok3, pok4, pok5, pok6, pok7 = proof
#     x, D, D1, D2 = pok1
#     x1 = np.mod(H(g1**D*h1**D*C**(-x) ), n1)
#     x2 = np.mod(H(g**Dh**D2*cd**(-id)), n)
#     if x != H(concat(x1, x2)):
#         return false
#     xe, De, De1, De2 = pok2
#     xe1 = np.mod(cB**De *h**De1 * c**(-xe), n)
#     xe2 = np.mod(g**D3 * h**De2 *cid**(-xe), n)
#     if xe != H(concat(xe1, xe2)):
#         return false

#     c, E1, E2, E3, E4, E5, E6 = pok3
#     #TODO: these should have mod's after them?
#     c1 = setpk**E1 * g**E2 * h**E6*(ce*g)**(-c)
#     c2 = g**E1 * h**E4 *ca**(-c)
#     c3 = g**E2 *h**E5*cz**(-c)
#     c4 = cB**E3 * h**E6*ce**(-c)
#     if c != H(concat(c1, c2, c3, c4)):
#         return false
#     e, u1, u, v1, v2, v3 = pok4
#     e1 = g**u1 * h**v1 *C1**(-e)
#     e2 = g**u *h**v2 * C2**(-e)
#     e3 = C1**u *h**v3 * C3**(-e)
#     if e != H(concat(e1, e2, e3)):
#         return false
