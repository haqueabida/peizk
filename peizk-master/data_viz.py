# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:27:50 2020
In this file we go through all the data.
You should time the creation of proofs and witnesses,
 and report the mean, median, and std err (std dev / N)
 of all the times and  also the sizes of the witnesses

@author: Abida
"""


import pandas as pd
import matplotlib.pyplot as plt

#from sklearn import metrics
import statistics
import math
#import sys
import scipy.stats

# Read the file of created id's

#enrolled1, enrolled_update_prove50, enrolled_update_prove, provetime, revoke_upmsg, verifytime

enrolled1 = pd.read_csv('enrolled1.csv')

#header id, a, B

#iterate on this csv file, either write to it or read to it.

provetimedf = pd.read_csv('provetime.csv')
#header id, proveTime


enrolled_update_prove = pd.read_csv('enrolled_update_prove.csv')
#header id, updateTime, proveTime
enrolled_update_prove.astype({'Proof_Time': 'float64'}).dtypes


enrolled_update_prove50 = pd.read_csv('enrolled_update_prove50.csv')
#header id, updateTime, proveTime

revoke_upd = pd.read_csv('revoke_upmsg.csv')


verifytimedf = pd.read_csv('verifytime.csv')
#################

#################

def utf8len(s):
    return len(s.encode('utf-8'))


def sizeof(x):
    """
    TODO: placeholder until we figure a better way to do sizes.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #return x.bit_length()
    return  (x.bit_length() + 7) // 8


def sizelistelmts(mylist, typech=int):
    list1 = list(map(typech, mylist))
    return list(map(sizeof, list1))


def size_of_signature(sigvals):
    spl_sig = list(map(lambda x: x[1:-1].split(', '), sigvals))
    
    spl_sig2 = [ sum([sizeof(int(elmt[0][1:])), sizeof(int(elmt[1][:-1])), utf8len(elmt[2])]) for elmt in spl_sig]

    return spl_sig2


def size_of_witness(df):
    """
    creates a dataframe to save size of witness (a,B)

    Parameters
    ----------
    df : pandas dataframe
        DESCRIPTION.

    Returns
    -------
    df2 : pandas.dataframe
        The input dataframe with the size of each witness appended as 2 columns

    """


    list_a = df['a']#.values.tolist()
    list_B =df['B']#.values.tolist()
    #regimentNamesCapitalized_m = list(map(capitalizer, regimentNames));
    #regimentNamesCapitalized_m

    list_a1 = sizelistelmts(list_a)
    list_B1 = sizelistelmts(list_B)

    df2 = df
    df2['size_a'] = list_a1
    df2['size_B'] = list_B1
    total =df
    total['witness_size'] = df2['size_a']+df2['size_B']

    return df2, total


def size_of_ID(df):
    list_id1 = df['ID']
    split_ID = strtuple_to_ints(list_id1)
    
    size_ID = [sum(sizelistelmts(r)) for r in split_ID]

    
    return size_ID



def size_of_commitments(df):
    """
    Peels out size of each of the commitments
    """

    C_a_vals = sizelistelmts(df['C_a'])
    C_id_vals = sizelistelmts(df['C_id'])
    C_B_vals = sizelistelmts(df['C_B'])
    C_z_vals =sizelistelmts( df['C_z'])
    C_e_vals = sizelistelmts(df['C_e'])
    C_w_vals = sizelistelmts(df['C_w'])

    df2 = df
    df2['C_a_size'] = C_a_vals
    df2['C_id_size']= C_id_vals
    df2['C_B_size']= C_B_vals
    df2['C_z_size']= C_z_vals
    df2['C_e_size']= C_e_vals
    df2['C_w_size'] = C_w_vals

    return df2


def strtuple_to_ints(mylist):
    res = list(map(lambda x: x[1:-1].split(', '), mylist))
    res2 = [list(map(int, r)) for r in res]

    return res2

def size_of_proofs(df):
    """
    Breaks up each proof and figures out the size.
    Each proof is a tuple.
    """

    pok2_vals = df['pok2']
    pok3_vals = df['pok3']
    pok4_vals = df['pok4']
    pok5_vals = df['pok5']
    pok6_vals = df['pok6']
    # remove parentheses and split into tuple

    #convert each element into an int
    spl_pok2 = strtuple_to_ints(pok2_vals)
    spl_pok3 = strtuple_to_ints(pok3_vals)
    spl_pok4 = strtuple_to_ints(pok4_vals)
    spl_pok5 = strtuple_to_ints(pok5_vals)
    spl_pok6 = strtuple_to_ints(pok6_vals)

    size_pok2 = [sum(sizelistelmts(r)) for r in spl_pok2]
    size_pok3 = [sum(sizelistelmts(r)) for r in spl_pok3]
    size_pok4 = [sum(sizelistelmts(r)) for r in spl_pok4]
    size_pok5 = [sum(sizelistelmts(r)) for r in spl_pok5]
    size_pok6 = [sum(sizelistelmts(r))  for r in spl_pok6]

    tot_size =  [sum(x) for x in zip(size_pok2, size_pok3,
                                     size_pok4, size_pok5, size_pok6)]


    df2 = df
    df['proof_size'] = tot_size
    df['pok2_size'] = size_pok2
    df['pok3_size'] = size_pok3
    df['pok4_size'] = size_pok4
    df['pok5_size'] = size_pok5
    df['pok6_size'] = size_pok6

    return df2





def bench_stats(df, colName):
    """
    returns the median, mean, and stdev of a df column of numbers

    Parameters
    ----------
    df : pandas dataframe
        DESCRIPTION.
    colName : string
        name of the column

    Returns
    -------
    df_med : TYPE
        DESCRIPTION.
    df_mean : TYPE
        DESCRIPTION.
    df_stdev : TYPE
        DESCRIPTION.

    """
    df_med = statistics.median(df[colName])
    df_mean = statistics.mean(df[colName])
    df_stdev = statistics.stdev(df[colName])
    df_skew = scipy.stats.skew(df[colName])

    return df_med, df_mean, df_stdev, df_skew


def	main():
    pr1_md, pr1_mean, pr1_stdev, pr1_skew = bench_stats(provetimedf, 'Proof_Time')
    pr2_md, pr2_mean, pr2_stdev, pr2_skew = bench_stats(enrolled_update_prove, 'Proof_Time')
    pr3_md, pr3_mean, pr3_stdev, pr3_skew = bench_stats(enrolled_update_prove50, 'Proof_Time') #error here


    a,b,c, g = bench_stats(enrolled_update_prove, 'Update_Time')
    d,e,f, h = bench_stats(enrolled_update_prove50, 'Update_Time')

    e1_md, e1_mean, e1_stdev, e1_skew = bench_stats(enrolled1,'enroll_time')
    
    print("Statistics on Enroll Time")
    print("-"*80)
    print("Name, Median, Mean, stdev, skew")
    print('Enroll,', e1_md*1000, ",", e1_mean*1000, ",", e1_stdev*1000, e1_skew)
    print("-"*80)
    
    
    print("Statistics on Update Time")
    print("-"*80)
    print("Name, Median, Mean, stdev, skew")
    print('after 1,', a*1000, ",", b*1000, ",", c*1000, g*1000)
    print('after 100,', d*1000, ",", e*1000, ",", f*1000, h*1000)
    print("-"*80)
    print("Statistics on Proving Time")
    print("-"*80)

    print("Name, Median, Mean, stdev, skew")

    #print("First Prove Stats, ", pr1_md, ",", pr1_mean, ",", pr1_stdev)
    print("After 1 update , ", pr2_md*1000, ",", pr2_mean*1000, ",", pr2_stdev*1000, pr2_skew)
    print("After 100 updates, ", pr3_md*1000, ",", pr3_mean*1000, ",", pr3_stdev*1000, pr2_skew)


    print("*"*80)
    print("Statistics on Commitment Time")
    print("-"*80)

    print("Name, Median, Mean, stdev, skew")

    cr2_md, cr2_mean, cr2_stdev, cr2_skew = bench_stats(enrolled_update_prove, 'Commitment_Time')
    cr3_md, cr3_mean, cr3_stdev, cr2_skew = bench_stats(enrolled_update_prove50, 'Commitment_Time') #error here



    #print("First Prove Stats, ", pr1_md, ",", pr1_mean, ",", pr1_stdev)
    print("After 1 update , ", cr2_md*1000, ",", cr2_mean*1000, ",", cr2_stdev*1000, cr2_skew)
    print("After 100 updates, ", cr3_md*1000, ",", cr3_mean*1000, ",", cr3_stdev*1000, cr2_skew)



    enrolled_with_sizes, witsize = size_of_witness(enrolled1)

  #  enra_md, enra_mean, enra_stdev = bench_stats(enrolled_with_sizes, 'size_a')
  #  enrB_md, enrB_mean, enrB_stdev = bench_stats(enrolled_with_sizes, 'size_B')
    w_md, w_mean, w_stdev, w_skew = bench_stats(enrolled1, 'witness_size')
    
    
    ID_sizes = size_of_ID(enrolled1)

    enrolled1['ID_size'] = ID_sizes
    
    id_md, id_mean, id_stdev, id_skew = bench_stats(enrolled1, 'ID_size')
    
    

    print("*"*80)

    print("Statistics on Witness/ID Size")
    print("-"*80)

    print("Name, Median, Mean, stdev, skew")

#    print("Size of a, ", enra_md, ",",enra_mean, ",",enra_stdev)
#    print("Size of B, ", enrB_md, ",",enrB_mean, ",",enrB_stdev)
    print("witness size,", w_md, ",",w_mean, ",",w_stdev, w_skew)
    print("ID size,", id_md, ",",id_mean, ",",id_stdev, id_skew)

    print("*"*80)
    print("Statistics on Proof Sizes")
    print("-"*80)

    print("Name , Median, Mean, stdev, skew")
    provetimedf1 = size_of_commitments(provetimedf)
    provetimedf1['Offline_Size']= provetimedf1.iloc[:, 14:20].sum(axis=1)

    # ca_mean, ca_med, ca_stdev = bench_stats(provetimedf1, 'C_a_size')
    # cid_mean, cid_med, cid_stdev = bench_stats(provetimedf1, 'C_id_size')
    # cB_mean, cB_med, cB_stdev = bench_stats(provetimedf1, 'C_B_size')
    # cz_mean, cz_med, cz_stdev = bench_stats(provetimedf1, 'C_z_size')
    # ce_mean, ce_med, ce_stdev = bench_stats(provetimedf1, 'C_e_size')
    # cw_mean, cw_med, cw_stdev = bench_stats(provetimedf1,'C_w_size')




    # print("Size of C_a,",  ca_mean, ",", ca_med, ",", ca_stdev)
    # print("Size of C_id,",  cid_mean, ",", cid_med, ",", cid_stdev)
    # print("Size of C_B,",  cB_mean, ",", cB_med, ",", cB_stdev)
    # print("Size of C_z,",  cz_mean, ",", cz_med, ",", cz_stdev)
    # print("Size of C_e,",  ce_mean, ",", ce_med, ",", ce_stdev)
    # print("Size of C_w,",  cw_mean, ",", cw_med, ",", cw_stdev)
    
    provetimedf2 = size_of_proofs(provetimedf)
    
   # provetimedf1['Online_Size']= provetimedf1.iloc[:, 23:29].sum(axis=1)


    # pf2_md, pf2_mean, pf2_stdev = bench_stats(provetimedf2, 'pok2_size')
    # pf3_md, pf3_mean, pf3_stdev = bench_stats(provetimedf2, 'pok3_size')
    # pf4_md, pf4_mean, pf4_stdev = bench_stats(provetimedf2, 'pok4_size')
    # pf5_md, pf5_mean, pf5_stdev = bench_stats(provetimedf2, 'pok5_size')
    # pf6_md, pf6_mean, pf6_stdev = bench_stats(provetimedf2, 'pok6_size')


    off_mean, off_med, off_stdev, off_skew = bench_stats(provetimedf1, 'Offline_Size')
    on_mean, on_med, on_stdev, on_skew = bench_stats(provetimedf1, 'proof_size')

    print("Offline Proof Size,",  off_mean, ",", off_med, ",", off_stdev, ",", off_skew)
    print("Online Proof Size,",  on_mean, ",", on_med, ",", on_stdev, ",", on_skew)



    # print("Size of pok2, ", pf2_md, ",",pf2_mean, ",",pf2_stdev)
    # print("Size of pok3, ", pf3_md, ",",pf3_mean, ",",pf3_stdev)
    # print("Size of pok4, ", pf4_md, ",",pf4_mean, ",",pf4_stdev)
    # print("Size of pok5, ", pf5_md, ",",pf5_mean, ",",pf5_stdev)
    # print("Size of pok6, ", pf6_md, ",",pf6_mean, ",",pf6_stdev)


##########################
    print("*"*80)
    print("Statistics on Verify") # and Commitment and Proof Times
    print("-"*80)

    print("Name , Median, Mean, stdev, skew")

   # comt_mean, comt_med, comt_stdev = bench_stats(provetimedf, 'Commitment_Time')
   # pt_mean, pt_med, pt_stdev = bench_stats(provetimedf, 'Proof_Time')
    
    # TODO: this needs to change -- we have a separate verifytime df now
    vert_mean, vert_med, vert_stdev, vert_skew = bench_stats(verifytimedf, 'verify_time')
    

    print("Verify Time,",  vert_mean*1000, ",", vert_med*1000, ",", vert_stdev*1000, vert_skew)
  #  print("Commitment Time,",  comt_mean*1000, ",", comt_med*1000, ",", comt_stdev*1000)
  #  print("Proof Time,",  pt_mean*1000, ",", pt_med*1000, ",", pt_stdev*1000)
    
    
    print("*"*80)
    print("Statistics on Revoke Times")
    print("-"*80)

    print("Name , Median, Mean, stdev, skew")

    revt_mean, revt_med, revt_stdev, revt_skew = bench_stats(revoke_upd, 'revoke_time')
    

    print("Revoked Time,",  revt_mean*1000, ",", revt_med*1000, ",", revt_stdev*1000, revt_skew)
    print("*"*80)
    
    
    revoke_upd['signature_size'] = size_of_signature(revoke_upd['Upmsg'])
    revsig_mean, revsig_med, revsig_stdev, revsig_skew = bench_stats(revoke_upd, 'signature_size')
    
    print("Signature Size,", revsig_mean, ",", revsig_med,",", revsig_stdev, ",", revsig_skew)
    print("*"*80)
    
    
    
    
    
    
    
    
##########################



if	__name__ == '__main__':
 	main()
