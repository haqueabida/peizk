import secrets
import random
import math
from ecdsa import SigningKey

from sigma import *
from helpfunctions import concat, generate_two_large_distinct_primes, hash_to_prime, bezoute_coefficients,\
    mul_inv, shamir_trick, calculate_product, hash_to_length, com_setup, generate_two_safe_primes

RSA_KEY_SIZE = 3072 # RSA key size for 128 bits of security (modulu size)
RSA_PRIME_SIZE = int(RSA_KEY_SIZE / 2)
# ACCUMULATED_PRIME_SIZE = 1  # taken from: LLX, "Universal accumulators with efficient nonmembership proofs", construction 1


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

def setup():
    """
    Outputs public parameters and initial accumulator.

    Returns
    -------
    L : set
        empty set of revoked identities.
    setpk : int
        the initial accumulator
    setsk : tuple
        factors of n (p,q) and initial randomness r.
    pp : tuple #TODO: fix this to remove pp_com?
        g,h,n,k (group elements, the public RSA modulus and length of modulus)

    """
    k = RSA_KEY_SIZE
    p,q = generate_two_safe_primes(k//2)
    # draw strong primes p,q
    # to replace by p,q = generate_two_safe_primes

    n = p*q

    # draw random number within range of [0,n-1]
    A0 = secrets.randbelow(n)
    A1 = secrets.randbelow(n)
    g = pow(A0,2,n)
    h = pow(A0,2,n)
    #setup for commitment scheme
    pp_com = com_setup()
    #parameters for RAMA
    sk = SigningKey.generate() # uses NIST192p
    vk = sk.verifying_key
    pp = (pp_com, g, h, n, k, vk) #TODO? can get rid of pp_com

    #Initialize blacklist accumulator
    pre_r = secrets.randbelow(n)
    r = hash_to_prime(pre_r, k)

    setsk = ((p-1)//2, (q-1)//2, r, sk)
    setpk = pow(g,r[0],n)
    L = {r}

    #To add signature in helper and here

    return (L, setpk, setsk, pp)


def enroll(pp, x, setsk, L):
    """ Enrolls a user by inputting their identity and returns their imei and bezout coefficients
    Parameters
    ----------
    pp: tuple
        public parameters g, h,n
    x : int
        identity
    setsk: tuple
        factors p' q' and random r
    L: set
        set of revoked identities
    """
    (pp_com, g, h, n, k, vk) = pp
    (p1, q1, r, sk) = setsk
    _id = hash_to_prime(x, k//2)
    while _id[0] % (p1*q1) == 1:
        _id = hash_to_prime(x, k//2)
    prodL = 1
    for x in L:
        # print (x)
        prodL = prodL * x[0]


    a1,b1 = bezoute_coefficients(prodL, _id[0])
    #ensuring that the cezout coefficient a is in the correct range
    k = 0
    if(a1 < 0):
	    while(a1 < pow(2,k//2) and a1 < 0):

	    	a1 = a1 + _id[0]
	    	k = k + 1
	    a = a1
	    b = b1 - k*prodL
    else:

    	while(a1 > 0 and a1 > pow(2,k//2)):
    		a1 = a1 - _id[0]
    		k = k + 1
    	a = a1
    	b = b1 + k*prodL
    b = -b
    B = pow_neg(g, b,n)
    # B = mul_inv(pow(g,abs(b),n),n)
    # print(a,B)
    return a,B, _id

def revoke(pp, setpk, L, _id, setsk):

    """
    Adds an identity to the list and updates the accumulator.
    Parameters
    ----------
    pp : TYPE
        group elements g, h, n,k
    setpk : TYPE
        current accumulator
    L : set
        set of revoked identities
    _id : int
        newly revoked identity
    msk : TYPE
        DESCRIPTION.

    Returns
    -------
    string
        when already revoked
    L, upmsg:
        updated set and signature of the revoked identity
    """

    (p1, q1, r, sk) = setsk
    (pp_com, g, h, n, k, vk) = pp
    if (_id in L):
        return "Already revoked"
    else:
        L.add(_id)
        setpk = pow(setpk, _id[0], n)

        upmsg = (setpk, _id[0])
        strupmsg = str(setpk) + str(_id[0])
        hupmsg = str(hash_to_length(int(strupmsg), RSA_KEY_SIZE))
        bhupmsg = bytes(hupmsg, encoding='utf-8')
        sig_upmsg = sk.sign(bhupmsg)
        return L,upmsg, sig_upmsg


def updCred(pp, setpk, upmsg, sig_upmsg,  skid, _id):
    """
    Phone side function

    Receives updated accumulator and signed message of revoked identity, updates the witness.
    Parameters
    ----------
    pp : tuple
        g, h, n
    setpk : int
        accumulator
    upmsg : TYPE
        signature
    skid : tuple
        Coefficients (a,B) (where log B = b and a,b are Bezout)
    _id : TYPE
        DESCRIPTION.

    Returns
    -------
    a1 : int
        Bezout coefficient 1
    B1 : int
        g^b where b is Bezout coefficient 2

    """
    (pp_com, g, h, n, k, vk) = pp
    (a,B) = skid
    (setpk1, _id1) = upmsg
    # print(_id1)
    # print(_id)
    strupmsg = str(setpk1) + str(_id1)
    hupmsg = str(hash_to_length(int(strupmsg), RSA_KEY_SIZE))
    bhupmsg = bytes(hupmsg, encoding='utf-8')
    if(vk.verify(sig_upmsg, bhupmsg) == 1):

    	(a0, r0) = bezoute_coefficients(_id1, _id)
    	a1 = (a*a0) % _id
    	r = (a1*_id1 - a)//_id
    	B1 = (pow_neg(setpk,r,n)*B)%n
    	return (a1,B1)
    else:
    	print("Error, signature not verified")

def prove(pp, setpk, skid, _id):
    """
    Phone side function

    Uses identity and witness to compute ZK proof.

    Parameters
    ----------
    pp : tuple
        g,h, n,k.
    setpk : int
        accumulator
    skid : tuple
        (a,B)
    _id : int
        identity

    Returns
    -------
    stmt : int
        setpk. #TODO: this will need to include tms soon
    proof : tuple
        output of proveAlgorithm.

    """

    (pp_com, g, h, n, k, vk) = pp
    (a,B) = skid
    tms = get_tms()
    stmt = (setpk, tms)
    witness = (skid, _id)
    proof1 = createComs(pp,stmt,witness)
    proof = proveAlgorithm(pp, stmt, proof1, witness)
    return stmt, proof

def createComs(pp,stmt,witness):
	(pp_com, g, h, n, k, vk) = pp
	setpkid, tms = stmt
	(skid, _id) = witness
	(a,B) = skid

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
	return C_a, C_id, C_B, C_z, C_e, C_w, w,r_id, r_a, r_w,r_e, r_z, r_e, z



def precomputeProofs(pp, stmt, proof1,  witness):
    (pp_com, g, h, n, k, vk) = pp
    setpkid, tms = stmt
    (skid, _id) = witness
    (a,B) = skid
    C_a, C_id, C_B, C_z, C_e, C_w, w,r_id, r_a, r_w,r_e, r_z, r_e,z  = proof1

    pre2 = pref1(C_e,C_id,_id,r_e,r_id,C_B,h,g,h,n,n,k, setpkid, tms)

    pre3 = pref2(C_e,C_a,C_z,C_B,_id,r_e,a,r_a,z,r_z,g,h,n,k, setpkid, tms)
    pre4 = pref3(C_z,C_w,C_id,r_z,w,r_w,_id,r_id,g,h,n, setpkid, tms)

    # pre5 = pref4b(C_id, _id, r_id, g, h, n,k)

    # pre6 = pref4b(C_a, a, r_a, g, h,n, k)
    return pre2, pre3, pre4



def proveAlgorithm(pp, stmt, proof1,  witness, preproofs):
    """
    AND- conjunction of 8 PoKs

    Parameters
    ----------
    pp : tuple
        g, h,n,k
    stmt : tuple
        accumulator and tms
    witness : tuple
        (a,B) (Bezout coefficient and power of Bezout coefficient)

    Returns
    -------
    proof : tuple
        commitments and outputs of PoKs (which are also tuples). #TODO: describe each item separately instead.

    """

    (pp_com, g, h, n, k, vk) = pp
    setpkid, tms = stmt
    (skid, _id) = witness
    (a,B) = skid
    C_a, C_id, C_B, C_z, C_e, C_w, w,r_id, r_a, r_w,r_e, r_z, r_e,z  = proof1

    pre2, pre3, pre4 = preproofs
    # w = secrets.randbelow(n)
    # r_id = secrets.randbelow(n)
    # r_a = secrets.randbelow(n)
    # r_w = secrets.randbelow(n)
    # r_z = secrets.randbelow(n)
    # r_e = secrets.randbelow(n)

    # C_id = commitment(_id,r_id,g,h,n)
    # C_a = commitment(a,r_a,g,h,n)
    # C_B = B*pow(g,w,n)%n
    # C_w = commitment(w,r_w,g,h,n)
    # z = _id*w
    # C_z = commitment(z,r_z,g,h,n)
    # C_e = commitment(_id,r_e,C_B,h,n)





    pok2 = f1(C_e,C_id,_id,r_e,r_id,C_B,h,g,h,n,n,k, setpkid, tms, pre2)
    # f1(C1, C2, x, r1, r2, g1, h1, g2, h2, n1, n2, RSA_PRIME_SIZE)
    pok3 = f2(C_e,C_a,C_z,C_B,_id,r_e,a,r_a,z,r_z,g,h,n,k, setpkid, tms, pre3)
    pok4 = f3(C_z,C_w,C_id,r_z,w,r_w,_id,r_id,g,h,n, setpkid, tms,pre4)
    # print("Id: ", _id)
    #tests to see if id is positive
    pok5 = f4b(C_id, _id, r_id, g, h, n,k)
    #checks to see if C_a > -2^k//2
    # print("a = ", a)
    pok6 = f4b(C_a, a, r_a, g, h,n, k)
    #tests to see if 2^l - id is positive
    # print("here")
    # C_idinv = mul_inv(C_id, n)
    # new_val = (pow(g,pow(2,k//2), n)*C_idinv)%n
    # print("new_val = ", new_val )
    # pok7 = f4(new_val, pow(2,k//2)-_id, -r_id,g,h,n,k)
    #pok8 = f4(pow(g,pow(2,k), n)//C_a, pow(2,k )- a, r_a,g,h,n,k)
    #proof = C_a, C_id, C_B, C_z, C_e, C_w, pok2, pok3, pok4, pok5, pok6, pok7, pok8
    proof = C_a, C_id, C_B, C_z, C_e, C_w, pok2, pok3,pok4, pok5, pok6 #pok5, pok7
    return proof

def zkverify(pp, stmt, proof):
    '''
        """
    Verifies the proof wrt statement

    Parameters
    ----------
    pp : tuple
        g,h,n, #TODO update this
    stmt : int
        same as accumulator? #TODO: add tms
    proof : tuple
        output of proveAlgorithm.

    Returns
    -------
    bool
        whether proof verifies.

    '''


    (pp_com, g, h, n, k, vk) = pp
    C_a, C_id, C_B, C_z, C_e, C_w, pok2, pok3, pok4, pok5, pok6 = proof # pok5,
    #C_a, C_id, C_B, C_z, C_e, C_w, pok2, pok3,pok4 = proof
    setpkid, tms = stmt

    vertms = get_tms()

    if (diff_tms(tms, vertms) == False):
        return False


    ch, D, D1, D2 = pok2
    ver2 = verify_f1(ch, C_e, C_id, D, D1, D2, C_B, h,g,h,n,n,k, setpkid, tms)

    c, D1, D2, D3, D4, D5, D6 = pok3
    ver3 = verify_f2(c, C_B, C_e,C_z,C_a, D1, D2, D3, D4, D5, D6,  g,h,n, k, setpkid, tms)

    e, u1, u, v1, v2, v3 = pok4
    ver4 = verify_f3(C_z, C_w, C_id, e, u1, u, v1,v2,v3, g,h,n, setpkid, tms)

    # print(ver3)

    e, z1, z2 = pok5
    ver5 = verify_f4b(C_id, g, h, n, k, e, z1, z2)

    e,z1,z2 = pok6
    ver6 = verify_f4b(C_a, g, h, n, k, e, z1, z2)


    # c, c2, alpha, beta, gamma = pok7

    # C_idinv = mul_inv(C_id, n)
    # new_val= (pow(g,pow(2,k//2), n)*C_idinv)%n
    # print("new_val = ", new_val)
    # ver7 = verify_f4(new_val, c, c2,alpha, beta, gamma, g,h,n)

#     c, c2, alpha, beta, gamma = pok7
#     v8 = verify_f4(pow(g,pow(2,l), n)//C_a, c, c2,alpha, beta, gamma, g,h,n)

    #return v2 and v3 and v4 and v5 #and v6 and v7 and v8
    print(ver5)
    return  ver3 and ver2 and ver4 and ver5 and ver6#and ver5
