import secrets
import random
import math
import numpy as np
from ecdsa import SigningKey
import time


from helpfunctions import hash_to_prime, is_prime, shamir_trick, generate_two_large_distinct_primes, mul_inv
from interface import setup, enroll,revoke, updCred, prove, zkverify

from unittest import TestCase
import unittest
from sigma import *

RSA_KEY_SIZE = 256  # RSA key size for 128 bits of security (modulu size)
RSA_PRIME_SIZE = int(RSA_KEY_SIZE / 2)
# ACCUMULATED_PRIME_SIZE = 128  # taken from: LLX, "Universal accumulators with efficient nonmembership proofs", construction 1


def create_list(size):
        res = []
        for i in range(size):
                x = secrets.randbelow(pow(2, 256))
                res.append(x)
        return res
    
# class SigmaTest(TestCase):
    # RSA_KEY_SIZE = 3072 
    # RSA_PRIME_SIZE = int(RSA_KEY_SIZE / 2)
    # p, q = generate_two_large_distinct_primes(RSA_PRIME_SIZE)
    # n = p*q
    # #TODO: is this how you choose g and h? also msg, r
    # g = secrets.randbelow(n)
    # h = secrets.randbelow(n)
    
    # #TODO: these tests are right now only for completeness, we need to show soundness also (cheating proofs)
    
    # def test_f0(self):
    #     msg = secrets.randbelow(pow(2, 256))
    #     r = secrets.randbelow(pow(2, 256))
    #     C = commitment(msg, r, g,h,n)       


    #     s1, s2 = f0(C, msg, r, g, h, n)

    #     is_valid = verify_f0(C, R, s1, s2, g, h,n)
    #     self.assertTrue(is_valid)
        
    #     #f2(setpk, ce, ca, cz, cB, my_id, re, z, rz, g,h,n,k):
    # def test_f2(self):
    #     #TODO: how to make up setpk? e, a, z,B
    #     setpk = secrets.randbelow(n)
    #     e = secrets.randbelow(n)
    #     a = secrets.randbelow(n)
    #     z = secrets.randbelow(n)
    #     B = secrets.randbelow(n)
    #     my_id = secrets.randbelow(n)
    #     ce= commitment(e, r, g,h,n)
    #     c, D1, D2, D3, D4, D5, D6 = f2(setpk, ce, ca, cz, cB, my_id, re, z, rz, g,h,n,k)
    #     is_valid = verify_f2(c, D1, D2, D3, D4, D5, D6)
    #     self.assertTrue(is_valid)
        
        
    # def test_f3(self):
    #     #Pick x,y,z to commit to 
    #     x = secrets.randbelow(pow(2,256))
    #     y = secrets.randbelow(pow(2,256))
    #     z = np.mod(x, y, n)
    #     rx = secrets.randbelow(pow(2,256))
    #     ry = secrets.randbelow(pow(2,256))
    #     rz = secrets.randbelow(pow(2,256))
    #     C1 = commitment(x,rx, g,h,n)
    #     C2 = commitment(x,rx, g,h,n)
    #     C3 = commitment(z,rz, g,h,n)
    #     e, u1, u, v1, v2, v3 = f3(C3, C1, C2, rz, z, x, rx, y, ry, g,h,n)


    #     is_valid = verify_f3(C3, C2, C1, e, u1, u, v1, v2, v3)
    #     self.assertTrue(is_valid)
        
class AccumulatorTest(TestCase):

        def test_overall(self): 

                #call setup 

                (L, setpk, setsk, pp) = setup()
                (p1, q1, r1, sk)  =  setsk
                (pp_com, g, h, n, k, vk) = pp

                #enroll a few parties, defined by ite
                enrolled = []
                credentials = []
                start_time = time.time()
                proofs = []

                ite = 10
                for i in range(ite):
                        x = secretsGenerator.randrange(1, n)
                        #_id = hash_to_prime(x, RSA_PRIME_SIZE)
                        a,B, _id  = enroll(pp,x,setsk,L)

                        enrolled.append(_id)
                        credentials.append((a,B))

                        
                        # print(a)

                        self.assertEqual(pow_neg(setpk,a,n),pow_neg(B,_id[0],n)*g%n)
                        
                        #check that enrolled parties can create proof that verifies
                        stmt, proof = prove(pp, setpk, (a,B), _id[0])
                        proofs.append((stmt,proof))

                        # print("--- %s seconds ---" % (time.time() - start_time))

                
                for i in proofs:
                    (stmt, proof) = i
                    self.assertTrue(zkverify(pp,stmt, proof))
                    # print("--- %s seconds ---" % (time.time() - start_time))



                

                #revoke one of the parties
                L, upmsg, sig_upmsg = revoke(pp, setpk, L, enrolled[0], setsk)
                # print(len(L))
                for i in range(1, ite):
                        #do update credential for the other
                        # print(credentials[i])
                        credentials[i] = updCred(pp, setpk, upmsg,sig_upmsg, credentials[i], enrolled[i][0])
                        # print("-------------------------******************-----------------------------------------------")
                        # print(credentials[i])
                        setpk1, temp = upmsg
                        # print(setpk!=setpk1)
                        #check that parties that are not revoked can create proof that verifies
                        stmt, proof = prove(pp, setpk1, credentials[i], enrolled[i][0])
                        val = zkverify(pp,stmt, proof)
                        self.assertTrue(val)
                
                setpk, temp = upmsg
                a,B = credentials[0]
                self.assertNotEqual(pow_neg(setpk,a,n),pow_neg(B,enrolled[0][0],n)*g%n)

                #check that parties that are revoked cannot create a proof that verifies
                setpk, temp = upmsg
                stmt, proof = prove(pp, setpk, credentials[0], enrolled[0][0])
                self.assertFalse(zkverify(pp,stmt, proof))

        # def test_setup(self):
        #         (L, setpk, setsk, pp) = setup()

        #         (p1, q1, r1)  =  setsk
        #         (pp_com, g, h, n)=pp
        #         self.assertEqual((2*p1+1)*(2*q1+1),n)
        # def test_enroll(self):
        #         (L, setpk, setsk, pp) = setup()
        #         (p1, q1, r1)  =  setsk
        #         (pp_com, g, h, n)=pp
        #         x = secrets.randbelow(pow(2, 256))
        #         _id = hash_to_prime(x, 1536)
        #         a, B = enroll(pp, x, setsk, L)
        #         if a < 0:
        #                 print("THIS CASE")
        #                 setpk = mul_inv(setpk,n)
        #                 a = -a
        #         self.assertEqual(pow(setpk,a,n),pow(B,_id[0],n)*g%n)
                



        # def test_revoke(self):
        #         (L, setpk, setsk, pp) = setup()
        #         (p1, q1, r1)  =  setsk
        #         (pp_com, g, h, n)=pp
        #         x = secrets.randbelow(pow(2, 256))
        #         _id = hash_to_prime(x, 1536)
        #         a, B = enroll(pp, x, setsk, L)
        #         L, upmsg = revoke(pp, setpk, L, _id, 0) #need to do signature
        #         (u1,u2) = upmsg
        #         if a < 0:
        #                 print("THIS CASE")
        #                 u1 = mul_inv(u1,n)
        #                 a = -a
        #         self.assertNotEqual(pow(u1, a, n), pow(B,u2,n)*g %n)
                
        
        # def test_updcred(self):
        #         (L, setpk, setsk, pp) = setup()
        #         (p1, q1, r1)  =  setsk
        #         print(r1)
        #         (pp_com, g, h, n)=pp
        #         x = secrets.randbelow(pow(2, 256))
        #         _id = hash_to_prime(x, 1536)
        #         a, B = enroll(pp, x, setsk, L)
        #         # L.add(_id)
        #         x1 = secrets.randbelow(pow(2, 256))
        #         _id1 = hash_to_prime(x1, 1536)
        #         # L.add(_id1)
        #         L1, upmsg = revoke(pp, setpk, L, _id1, 0)
        #         # print (upmsg)
        #         print(L1)
        #         (setpk1, skid) = upmsg
        #         (a1,B1) = updCred(pp, setpk, upmsg, (a,B), _id[0])
                
        #         if a1 < 0:
        #                 print("THIS CASE")
        #                 setpk1 = mul_inv(setpk1,n)
        #                 a1 = -a1
        #         self.assertEqual(pow(setpk1,a1,n),pow(B1,_id[0],n)*g%n)


        # def test_hash_to_prime(self):
        #         x = secrets.randbelow(pow(2, 256))
        #         h, nonce = hash_to_prime(x, 128)
        #         self.assertTrue(is_prime(h))
        #         self.assertTrue(h, math.log2(h) < 128)

        # def test_add_element(self):
        #         n, A0, S = setup()

        #         x0 = secrets.randbelow(pow(2, 256))
        #         x1 = secrets.randbelow(pow(2, 256))

        #         # first addition
        #         A1 = add(A0, S, x0, n)
        #         nonce = S[x0]

        #         proof = prove_membership(A0, S, x0, n)
        #         self.assertEqual(len(S), 1)
        #         self.assertEqual(A0, proof)
        #         self.assertTrue(verify_membership(A1, x0, nonce, proof, n))

        #         # second addition
        #         A2 = add(A1, S, x1, n)
        #         nonce = S[x1]

        #         proof = prove_membership(A0, S, x1, n)
        #         self.assertEqual(len(S), 2)
        #         self.assertEqual(A1, proof)
        #         self.assertTrue(verify_membership(A2, x1, nonce, proof, n))

        #         # delete
        #         A1_new = delete(A0, A2, S, x0, n)
        #         proof = prove_membership(A0, S, x1, n)
        #         proof_none = prove_membership(A0, S, x0, n)
        #         self.assertEqual(len(S), 1)
        #         self.assertEqual(proof_none, None)
        #         self.assertTrue(verify_membership(A1_new, x1, nonce, proof, n))

        # def test_proof_of_exponent(self):
        #         # first, do regular accumulation
        #         n, A0, S = setup()
        #         x0 = secrets.randbelow(pow(2, 256))
        #         x1 = secrets.randbelow(pow(2, 256))
        #         A1 = add(A0, S, x0, n)
        #         A2 = add(A1, S, x1, n)

        #         Q, l_nonce, u = prove_membership_with_NIPoE(A0, S, x0, n, A2)
        #         is_valid = verify_exponentiation(Q, l_nonce, u, x0, S[x0], A2, n)
        #         self.assertTrue(is_valid)

        # def test_proof_of_nizk(self):
        #         RSA_KEY_SIZE = 3072 
        #         RSA_PRIME_SIZE = int(RSA_KEY_SIZE / 2)
        #         p, q = generate_two_large_distinct_primes(RSA_PRIME_SIZE)
        #         n = p*q
        #         u = secrets.randbelow(pow(2, 256))
        #         x = secrets.randbelow(pow(2, 256))
        #         w = pow(u,x,n)
        #         g = secrets.randbelow(n)
        #         h = secrets.randbelow(n)
        #         l,z,Q_g, Q_u, r_x, r_p = nizkprove_exponentiation(g,h,u,x,w,n)

        #         is_valid = verify_nizk(l,z,Q_g, Q_u, r_x, r_p , u, w, g, h, n)
        #         self.assertTrue(is_valid)

        # def test_batch_add(self):
        #         n, A0, S = setup()

        #         elements_list = create_list(10)

        #         A_post_add, nipoe = batch_add(A0, S, elements_list, n)
        #         self.assertEqual(len(S), 10)

        #         nonces_list = list(map(lambda e: hash_to_prime(e)[1], elements_list))
        #         is_valid = batch_verify_membership_with_NIPoE(nipoe[0], nipoe[1], A0, elements_list, nonces_list, A_post_add, n)
        #         self.assertTrue(is_valid)

        # def test_batch_proof_of_membership(self):
        #         n, A0, S = setup()

        #         elements_list = create_list(10)

        #         A = A0
        #         for x in elements_list:
        #                 A = add(A, S, x, n)
        #         A_final = A

        #         elements_to_prove_list = [elements_list[4], elements_list[7], elements_list[8]]
        #         A_intermediate = batch_prove_membership(A0, S, elements_to_prove_list, n=n)
        #         nonces_list = list(map(lambda e: hash_to_prime(e)[1], elements_to_prove_list))
        #         is_valid = batch_verify_membership(A_final, elements_to_prove_list, nonces_list, A_intermediate, n)
        #         self.assertTrue(is_valid)

        # def test_batch_proof_of_membership_with_NIPoE(self):
        #         n, A0, S = setup()

        #         elements_list = create_list(10)

        #         A = A0
        #         for x in elements_list:
        #                 A = add(A, S, x, n)
        #         A_final = A

        #         elements_to_prove_list = [elements_list[4], elements_list[7], elements_list[8]]
        #         Q, l_nonce, u = batch_prove_membership_with_NIPoE(A0, S, elements_to_prove_list, n, A_final)
        #         nonces_list = list(map(lambda e: hash_to_prime(e)[1], elements_to_prove_list))
        #         is_valid = batch_verify_membership_with_NIPoE(Q, l_nonce, u, elements_to_prove_list, nonces_list, A_final, n)
        #         self.assertTrue(is_valid)

        # def test_shamir_trick_1(self):
        #         n = 23
        #         A0 = 2

        #         prime0 = 3
        #         prime1 = 5

        #         A1 = pow(A0, prime0, n)
        #         A2 = pow(A1, prime1, n)

        #         proof0 = pow(A0, prime1, n)
        #         proof1 = pow(A0, prime0, n)

        #         agg_proof = shamir_trick(proof0, proof1, prime0, prime1, n)
        #         power = pow(agg_proof, prime0 * prime1, n)

        #         is_valid = power == A2
        #         self.assertTrue(is_valid)

        # def test_shamir_trick_2(self):
        #         n, A0, S = setup()

        #         elements_list = create_list(2)

        #         A1 = add(A0, S, elements_list[0], n)
        #         A2 = add(A1, S, elements_list[1], n)

        #         prime0 = hash_to_prime(elements_list[0], nonce=S[elements_list[0]])[0]
        #         prime1 = hash_to_prime(elements_list[1], nonce=S[elements_list[1]])[0]

        #         proof0 = prove_membership(A0, S, elements_list[0], n)
        #         proof1 = prove_membership(A0, S, elements_list[1], n)

        #         agg_proof = shamir_trick(proof0, proof1, prime0, prime1, n)

        #         is_valid = pow(agg_proof, prime0 * prime1, n) == A2
        #         self.assertTrue(is_valid)

        # def test_prove_non_membership(self):
        #         n, A0, S = setup()

        #         elements_list = create_list(3)

        #         A1 = add(A0, S, elements_list[0], n)
        #         A2 = add(A1, S, elements_list[1], n)
        #         A3 = add(A2, S, elements_list[2], n)

        #         proof = prove_non_membership(A0, S, elements_list[0], S[elements_list[0]], n)
        #         self.assertIsNone(proof)

        #         x = create_list(1)[0]
        #         prime, x_nonce = hash_to_prime(x)
        #         proof = prove_non_membership(A0, S, x, x_nonce, n)
        #         is_valid = verify_non_membership(A0, A3, proof[0], proof[1], x, x_nonce, n)
        #         self.assertTrue(is_valid)

                 
        # def test_batch_delete(self):
        #         n, A0, S = setup()

        #         elements_list = create_list(5)

        #         A = A0
        #         for i in range(len(elements_list)):
        #                 A = add(A, S, elements_list[i], n)
        #         A_pre_delete = A

        #         elements_to_delete_list = [elements_list[0], elements_list[2], elements_list[4]]
        #         nonces_list = list(map(lambda e: hash_to_prime(e)[1], elements_to_delete_list))

        #         proofs = list(map(lambda x: prove_membership(A0, S, x, n), elements_to_delete_list))

        #         A_post_delete, nipoe = batch_delete_using_membership_proofs(A_pre_delete, S, elements_to_delete_list, proofs, n)

        #         is_valid = batch_verify_membership_with_NIPoE(nipoe[0], nipoe[1], A_post_delete, elements_to_delete_list, nonces_list, A_pre_delete, n)
        #         self.assertTrue(is_valid)

        # def test_create_all_membership_witnesses(self):
        #         n, A0, S = setup()

        #         elements_list = create_list(3)

        #         A1, nipoe = batch_add(A0, S, elements_list, n)
        #         witnesses = create_all_membership_witnesses(A0, S, n)

        #         elements_list = list(S.keys())  # this specific order is important
        #         for i, witness in enumerate(witnesses):
        #                 self.assertTrue(verify_membership(A1, elements_list[i], S[elements_list[i]], witness, n))

        # def test_agg_mem_witnesses(self):
        #         n, A0, S = setup()

        #         elements_list = create_list(3)

        #         A1, nipoe = batch_add(A0, S, elements_list, n)
        #         witnesses = create_all_membership_witnesses(A0, S, n)

        #         elements_list = list(S.keys())  # this specific order is important
        #         for i, witness in enumerate(witnesses):
        #                 self.assertTrue(verify_membership(A1, elements_list[i], S[elements_list[i]], witness, n))

        #         nonces_list = [S[x] for x in elements_list]
        #         agg_wit, nipoe = aggregate_membership_witnesses(A1, witnesses, elements_list, nonces_list, n)

        #         is_valid = batch_verify_membership_with_NIPoE(nipoe[0], nipoe[1], agg_wit, elements_list, nonces_list, A1, n)
        #         self.assertTrue(is_valid)


if __name__ =='__main__':
    unittest.main()