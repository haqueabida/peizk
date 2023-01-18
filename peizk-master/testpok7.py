# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:52:13 2020

@author: Abida
"""



(L, setpk, setsk, pp) = setup()
msk = 1
(p1, q1, r1)  =  setsk
(pp_com, g, h, n, k) = pp

#enroll a few parties, defined by ite
enrolled = []
credentials = []
ite = 3
for i in range(ite):
        x = secretsGenerator.randrange(1, n)
        #_id = hash_to_prime(x, RSA_PRIME_SIZE)
        a,B, _id  = enroll(pp,x,setsk,L)
        enrolled.append(_id)
        credentials.append((a,B))
                        


    
(pp_com, g, h, n, k) = pp
#(skid, _id) 
#(a,B) = skid
_id = _id[0]
w = secrets.randbelow(n)
r_id = secrets.randbelow(n)
r_a = secrets.randbelow(n)
r_w = secrets.randbelow(n)
r_z = secrets.randbelow(n)
r_e = secrets.randbelow(n)

C_id = commitment(_id,r_id,g,h,n)
C_a = commitment(a,r_a,g,h,n)
C_B = B*pow(g,w,n)%n
C_w = commitment(w,r_w,g,h,n)
z = _id*w
C_z = commitment(z,r_z,g,h,n)
C_e = commitment(_id,r_e,C_B,h,n)


C_idinv = mul_inv(C_id, n)
new_val = (pow(g,pow(2,k//2), n)*C_idinv)%n
print("new_val = ", new_val )
pok7 = f4(new_val, pow(2,k//2)-_id, -r_id,g,h,n,k)

c, c2, alpha, beta, gamma = pok7

C_idinv = mul_inv(C_id, n)
new_val= (pow(g,pow(2,k//2), n)*C_idinv)%n
print("new_val = ", new_val)
ver7 = verify_f4(new_val, c, c2,alpha, beta, gamma, g,h,n)

####open up ver7
R, s1, s2 = alpha #because it is output of f0
t1 = verify_f0(new_val, R, s1, s2, g, h,n)

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

final_check = (new_val==(c2prod%n))

    #return t1 and t2 and t3 and final_check
