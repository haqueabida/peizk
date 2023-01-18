import secrets
import random
import math
from ecdsa import SigningKey
import time
import csv


from helpfunctions import hash_to_prime, is_prime, shamir_trick, generate_two_large_distinct_primes, mul_inv, get_tms
from interface import setup, enroll,revoke, updCred, prove, zkverify, createComs, proveAlgorithm

from unittest import TestCase
import unittest
from sigma import *

RSA_KEY_SIZE = 3072  # RSA key size for 128 bits of security (modulu size)
RSA_PRIME_SIZE = int(RSA_KEY_SIZE / 2)


def timertest():

	# k = RSA_KEY_SIZE
	# p,q = generate_two_safe_primes(k//2)
 #    # draw strong primes p,q
 #    # to replace by p,q = generate_two_safe_primes
	# print(p)
	# print(q)
	# n = p*q

 #    # draw random number within range of [0,n-1]
	# A0 = secrets.randbelow(n)
	# A1 = secrets.randbelow(n)
	# g = pow(A0,2,n)
	# h = pow(A0,2,n)
 #    #setup for commitment scheme
	# pp_com = com_setup()
 #    #parameters for RAMA
	# sk = SigningKey.generate() # uses NIST192p
	# vk = sk.verifying_key
	# pp = (pp_com, g, h, n, k, vk) #TODO? can get rid of pp_com

 #    #Initialize blacklist accumulator
	# pre_r = secrets.randbelow(n)
	# r = hash_to_prime(pre_r, k)
    
	# setsk = ((p-1)//2, (q-1)//2, r, sk)
	# setpk = pow(g,r[0],n)
	# L = {r}

 #    #To add signature in helper and here
	# p1 = (p-1)//2
	# q1 = (q-1)//2

	# print(r)
	# print(setpk)
	# print((p-1)//2)
	# print((q-1)//2)
	# print(pp_com)
	# print(g)
	# print(h)
	# print(n)
	# print(k)
	r = (4560057781571392056709216532792792580997420365459901921296101521129119610239443701442191856957279416436166189752829623995685105504031149702136436292050427223508068922808626009849897274026644267273497688475375885857809600779166784502039775772036154362447390695778825500681469063822906093194751895074359841884138259042312555699128271700468466180694585318228834659724449957827103946012139202872623788723739877723149821143425958203428342420210662507046401889322818850964609248497332108331905334637366240172695183702474682179917478057192252049101286396091463856499559515432982029236645913813727025204586198960345606014039329442847934828669989686090325227634194134407815733018033332889878633863220354378333691591155815834505266930358828638663104213023135626785666313432307604949276827758490435116600758573727184766154710350715915843535703916014872790791886531582044536557368468134381433426263105697146311742507809813950813363922353, 278)
	setpk = 29606457742385443671724482489650081231183397433414114231110816902929141216181883436033001368827640662546612030605011982214421345269760955669362107432482211780550312436031136728243405270401765813709397727708170201088920705927256723978790480050765423847940522429586390091940893669682735220602485032132961030468410095461104715804153320484363890003961313246634924129115617856861218838164688638058980035318545527031708496849836166914187560304203158733955667299490743
	p1 = 353682452495543407502262938025683810802196338491948036892308582087857389649115482412279844127320013766144079800648618501492813288233306491979774491990609654853172229897921213227358542798488704621873412380254787639894751784383030820
	q1 = 45416586472074118303043796209906100186716233237630888250276252321924843787819437608456758419852766374216615094244292338400643944799885552442998643878353640201709247271400382544540322045510092902675831820424015010160378604818512640
	pp_com =  (2797521862614265430207728006481100284359116355236162099068343189088318807122755591061993074524571719441409377255743079637926154143254802905100567075883461075661474395354676058295058949907217132291415587813519554875917748401604900310021817135139961187946774132440235551320413497302981277342167363755425617977720475082374833199834448006011942207655943580094325604822204464002524199570731167426339796577918673616648494391071118564821741631828757842719479982002842387004479011509773628515769620575891668806256169298019433331552242686731522156661271054698536010876624815482860983292404889994299555712266588950359158109469117468483996278791926795551741834391836828479419002173921663609400580769949884379571846951800024370642577248987935395718532182461020466844802414341099968525262003168736314977507092853310086509969837227817310735404689702191442798792931102116915488397971701524717528906767225298874316559462088686636729062498987, 1205291871378386766166242753271784738872432805648369420773908996426604600648521824188394972083296702389048822381162218567103851111679134219631115543231592824456120699709467065527188135707052509055913008063186950814157946760025933244581342137140059980662359198946111815448429558630228919888335580139389825021299227580821643432593925623296750850118691563156859947259022317706155213003437593680180924719814573177547589001951479854003326149803073764575633145172502228974360514920206793189309418245099123337954105586850518613231512760546672084060215926444873453829846922641045492626609839435766191585067788855227057414659105547206044852278310668299320887104680138918871451051178784918398852337948245381212916163488955182867263630680057698188018615919833662558257162993436573308424090781956393101902250664268368156334575268715573205693771714647927869658806134819726178039411302005727891340235973469813438335447023312796672874875584, 1205291871378386766166242753271784738872432805648369420773908996426604600648521824188394972083296702389048822381162218567103851111679134219631115543231592824456120699709467065527188135707052509055913008063186950814157946760025933244581342137140059980662359198946111815448429558630228919888335580139389825021299227580821643432593925623296750850118691563156859947259022317706155213003437593680180924719814573177547589001951479854003326149803073764575633145172502228974360514920206793189309418245099123337954105586850518613231512760546672084060215926444873453829846922641045492626609839435766191585067788855227057414659105547206044852278310668299320887104680138918871451051178784918398852337948245381212916163488955182867263630680057698188018615919833662558257162993436573308424090781956393101902250664268368156334575268715573205693771714647927869658806134819726178039411302005727891340235973469813438335447023312796672874875584)
	g = 39554309319134814807184540478536617756504558842802837876647055936517937502194823935872827785597109401772741615375506582159775565155879358805584124425820589048451024630684708035347912245953868119254997978361180299284803591294663852108992501521210843225066806678326680470959865854644082265296111708255108971559010812709239651528292016657377458397756345516353771232148725950066447577349466880809080130419310477059891611386216527035771030295373662227944919433107686
	h = 39554309319134814807184540478536617756504558842802837876647055936517937502194823935872827785597109401772741615375506582159775565155879358805584124425820589048451024630684708035347912245953868119254997978361180299284803591294663852108992501521210843225066806678326680470959865854644082265296111708255108971559010812709239651528292016657377458397756345516353771232148725950066447577349466880809080130419310477059891611386216527035771030295373662227944919433107686
	n =64252198749676374818605584015523000298970815993889155920325946563262337300428471790021435791915052122712854327000308382982834501640952203077428991970110891305740385989939377429033052772630403255844669753913364585287083293827012933103447358961795196223759985736973472074758170779564275055698543031204510308578105281564143953094345356746576616495661483010684345551703419852640046781711481213537011104994185469101173822342831177791233036749966457632823209121346121
	k =3072


	sk = SigningKey.generate()

	vk = sk.verifying_key
	pp = (pp_com, g, h, n, k, vk)
	
	setsk = (p1, q1, r, sk)
	# setpk = pow(g,r[0],n)
	L = {r}
	ite = 100
	# enrolled = []
	# credentials = []
	# #enroll 10000 imeis and time to enroll
	# fields = ["ID", "a", "B", "enroll_time"]
	# with open("enrolled1.csv", 'w') as csvfile:
	# 	csvwriter = csv.writer(csvfile)
	# 	csvwriter.writerow(fields)
	# for i in range(ite):
	# 	x = secretsGenerator.randrange(1, n)
 #        #_id = hash_to_prime(x, RSA_PRIME_SIZE)
	# 	start_time = time.time()
	# 	a,B, _id  = enroll(pp,x,setsk,L)
	# 	enroll_time = time.time() - start_time 
	# 	with open("enrolled1.csv", 'a+', newline='') as csvfile:
	# 		csvwriter = csv.writer(csvfile)
	# 		csvwriter.writerow([_id,a,B, enroll_time])

	# 	enrolled.append(_id)
	# 	credentials.append((a,B))




	enrolled = []
	credentials = []
	with open('enrolled1.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				# print(f'Column names are {", ".join(row)}')
				line_count += 1
			else:
				to_append = tuple(map(int,row[0][1:-1].split(",")))
				enrolled.append(to_append)

				credentials.append((int(row[1]), int(row[2])))
				# print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
				line_count += 1
		print(f'Processed {line_count} lines.')

	print(enrolled[0][0])
	print(type(enrolled[0][0]))

	# time to create 10000 proofs and verify 10000 proofs
	# fields = ["ID", "Commitment_Time", "Proof_Time", "Verify_time", "C_a", "C_id", "C_B", "C_z", "C_e", "C_w", "pok2", "pok3", "pok4", "pok5", "pok6"]
	# with open("provetime.csv", 'w') as csvfileP:
	# 	csvwriter = csv.writer(csvfileP)
	# 	csvwriter.writerow(fields)
	 
	# for i in range(len(enrolled)):
	# 	start_time = time.time()
	# 	(pp_com, g, h, n, k, vk) = pp
	# 	skid = credentials[i]
	# 	tms = get_tms()
	# 	stmt = (setpk, tms)
	# 	witness = (skid, enrolled[i][0])
	# 	start_time = time.time()
	# 	proof1 = createComs(pp, stmt, witness)
	# 	C_a, C_id, C_B, C_z, C_e, C_w, w,r_id, r_a, r_w,r_e, r_z, r_e, z = proof1
	# 	com_time = (time.time() - start_time)
	# 	start_time = time.time()
	# 	proof = proveAlgorithm(pp, stmt, proof1,  witness)
	# 	C_a, C_id, C_B, C_z, C_e, C_w, pok2, pok3,pok4, pok5, pok6 = proof
	# 	proof_time = (time.time() - start_time)
	# 	start_time = time.time()
	# 	V = zkverify(pp, stmt, proof)
	# 	verify_time = time.time() - start_time
	# 	# print((time.time() - start_time))
	# 	with open("provetime.csv", 'a+', newline='') as csvfileP:
	# 		csvwriter = csv.writer(csvfileP)


	# 		csvwriter.writerow([enrolled[i][0], com_time, proof_time, verify_time, C_a, C_id, C_B, C_z, C_e, C_w, pok2, pok3, pok4, pok5, pok6])
	# proof_time = (time.time() - start_time)
	# print("Time for 10000 proofs == ", proof_time)


	#time to do 1 update for 9999 parties

	# L, upmsg, sig_upmsg = revoke(pp, setpk, L, enrolled[0], setsk)
	# enrolled = enrolled[1:]
	# credentials = credentials[1:]
	# fields = ["ID", "Update_Time", "Commitment_Time", "Proof_Time"]
	# with open("enrolled_update_prove.csv", 'w') as csvfileU:
	# 	csvwriter = csv.writer(csvfileU)
	# 	csvwriter.writerow(fields)
	
	# for i in range(len(enrolled)): 
	# 	start_time = time.time()
	# 	credentials[i] = updCred(pp, setpk, upmsg,sig_upmsg, credentials[i], enrolled[i][0]) 
	# 	upd_time = (time.time() - start_time)
	# 	(pp_com, g, h, n, k, vk) = pp
	# 	skid = credentials[i]
	# 	tms = get_tms()
	# 	stmt = (setpk, tms)
	# 	witness = (skid, enrolled[i][0])
	# 	start_time = time.time()
	# 	setpk1, temp = upmsg
	# 	proof1 = createComs(pp, stmt, witness)
	# 	C_a, C_id, C_B, C_z, C_e, C_w, w,r_id, r_a, r_w,r_e, r_z, r_e, z = proof1
	# 	com_time = (time.time() - start_time)
	# 	start_time = time.time()
	# 	proof = proveAlgorithm(pp, stmt, proof1,  witness)
	# 	# stmt, proof = prove(pp, setpk1, credentials[i], enrolled[i][0])
	# 	proof_time = (time.time() - start_time)
	# 	# print((time.time() - start_time))
	# 	with open("enrolled_update_prove.csv", 'a+', newline='') as csvfileU:
	# 		csvwriter = csv.writer(csvfileU)
	# 		csvwriter.writerow([enrolled[i][0], upd_time, com_time, proof_time])
	

	#time to revoke 100 parties and update messages
	# fields = ["Revoked_ID", "revoke_time", "Upmsg"]
	# with open("revoke_upmsg.csv", 'w') as csvfileUR:
	# 	csvwriter = csv.writer(csvfileUR)
	# 	csvwriter.writerow(fields)
	k = 5
	revoked = []
	for i in range(k):
		start_time = time.time()
		L, upmsg, sig_upmsg = revoke(pp, setpk, L, enrolled[i], setsk)
		rev_time = (time.time() - start_time)
		revoked.append([upmsg, sig_upmsg])
		# with open("revoke_upmsg.csv", 'a+', newline='') as csvfileUR:
		# 	csvwriter = csv.writer(csvfileUR)
		# 	csvwriter.writerow([enrolled[i][0], rev_time, (upmsg,sig_upmsg)])
	#time to 100 updates for 10000 nodes 
	# fields = ["ID", "Update_Time", "Commitment_Time", "Proof_Time"]
	# with open("enrolled_update_prove100.csv", 'w') as csvfileU1:
	# 	csvwriter = csv.writer(csvfileU1)
	# 	csvwriter.writerow(fields)
	enrolled = enrolled[k:]
	credentials = credentials[k:]
	start_time = time.time()
	for i in range(len(enrolled)):
		print("Party", i+1)
		j = 1
		for revMsg in revoked:
			print("updating according to reveoked user number ", j )
			j+=1
			credentials[i] = updCred(pp, setpk, revMsg[0],revMsg[1], credentials[i], enrolled[i][0])

	upd_time = (time.time() - start_time)
	print("time taken for 10 updates by 90 parties", upd_time)
		# print((time.time() - start_time))
		# upmsg = revoked[-1][0]
		# (pp_com, g, h, n, k, vk) = pp
		# skid = credentials[i]
		# tms = get_tms()
		# stmt = (setpk, tms)
		# witness = (skid, enrolled[i][0])
		# start_time = time.time()
		# setpk1, temp = upmsg
		# proof1 = createComs(pp, stmt, witness)
		# C_a, C_id, C_B, C_z, C_e, C_w, w,r_id, r_a, r_w,r_e, r_z, r_e, z = proof1
		# com_time = (time.time() - start_time)
		# start_time = time.time()
		# proof = proveAlgorithm(pp, stmt, proof1,  witness)
		# # stmt, proof = prove(pp, setpk1, credentials[i], enrolled[i][0])
		# proof_time = (time.time() - start_time)
		# with open("enrolled_update_prove100.csv", 'a+', newline='') as csvfileU1:
		# 	csvwriter = csv.writer(csvfileU1)
		# 	csvwriter.writerow([enrolled[i][0], upd_time, com_time, proof_time])
	


	






timertest()
