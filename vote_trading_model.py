########################################################
# Bundling and Vote Trading of Legislative Quanta
########################################################

#############
# Model details
# In this early model, there are n voters and m quanta
# Each voter assigns a utility to each quanta, taking a value in {-2, -1, 1, 2}
# Compare the results of passing quanta by:
# 1) Bundling
# 2) Majority Rule
# 3) Utility Rule
# 4) Vote Trading
# Voters can trade their vote on a -1 utility quanta for someone else's vote
# on a +2 utility quanta. Two voters can also trade no votes if there are
# quanta they value at +1 and -2
# Value of a voting method will be measured by adding the utility of passed 
# quanta and subtracting the utility of failed quanta
#############

import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import random as rand
import json
import copy
import scipy.stats as st
import math
from sklearn.decomposition import PCA

def dist(a,b):
    val = 0
    for i in range(len(a)):
        val += (a[i]-b[i])**2
    return np.sqrt(val)

def vote_sim(n, m, method, probs):   
    utils = [-2,-1,1,2]
    
    '''Generate utilities for each of the voters and the utilities'''
    ##########
    # Method 1: Uniform and noisy
    ##########
    if method == 1:
        # probs = [0.25, 0.25, 0.25, 0.25]
        # probs = [0.2,0.35,0.2,0.25]
        # probs = [0.5,0.0,0.5,0.0]
        # probs = [0.0, 0.5, 0.5, 0.0]
        utilities = [rand.choices(utils, weights=probs, k=m) for _ in range(n)]

    ##########
    # Method 2: Spatial model of voting
    ##########
    elif method == 2:
        utilities = [[] for _ in range(n)]
        
        dim = 2
        voter_positions = [[rand.random() for _ in range(dim)] for _ in range(n)]
        for _ in range(m):
            sq_pos = np.asarray([rand.random() for _ in range(dim)])
            quantum_pos = np.asarray([rand.random() for _ in range(dim)])
            # Old method that uses perpendicular lines to line connecting sq and quantum
            for i in range(n):
                v_pos = np.asarray(voter_positions[i])
                if dist(v_pos, sq_pos) > dist(v_pos, quantum_pos):
                    temp = 2*quantum_pos - sq_pos
                    if dist(v_pos, temp)>dist(v_pos,sq_pos):
                        utilities[i].append(utils[2])
                    else:
                        utilities[i].append(utils[3])
                else:
                    temp = 2*sq_pos - quantum_pos
                    if dist(v_pos, temp)>dist(v_pos, quantum_pos):
                        utilities[i].append(utils[1])
                    else:
                        utilities[i].append(utils[0])
            
            # for i in range(n):
            #     v_pos = np.asarray(voter_positions[i])
            #     if dist(v_pos, sq_pos) > dist(v_pos, quantum_pos):
            #         if 3*dist(v_pos, sq_pos) > dist(v_pos, quantum_pos):
            #             utilities[i].append(utils[3])
            #         else:
            #             utilities[i].append(utils[2])
            #     else:
            #         if dist(v_pos, sq_pos) < 3*dist(v_pos, quantum_pos):
            #             utilities[i].append(utils[0])
            #         else:
            #             utilities[i].append(utils[1])
    
    ##########
    # Method 3: Good and Bad quanta with noisy signal
    ##########
    elif method == 3:
        probs1 = [0.1, 0.4, 0.1, 0.4]
        probs2 = [0.4, 0.1, 0.4, 0.1]
        
        utilities = [[] for _ in range(n)]
        for _ in range(m):
            if rand.random()<0.5:
                for i in range(n):
                    utilities[i].append(rand.choices(utils, weights=probs1, k=1)[0])
            else:
                for i in range(n):
                    utilities[i].append(rand.choices(utils, weights=probs2, k=1)[0])
                    
    else:
        print('method error')
        return None
    
    # print(np.sum(utilities))
    
    '''Bundled Voting'''
    # All quanta are placed in a bill and each voter supports the bill 
    # if it has a sum positive utility
    
    bill_contents = [1 for _ in range(m)]
    bill_votes = 0
    bill_util = 0
    for i in range(n):
        total_util = 0
        for j in range(m):
            total_util += bill_contents[j]*utilities[i][j]
        bill_util += total_util
        if total_util>0:
            bill_votes += 1
            
    # if bill_votes>n/2:
    #     print(f'The bundled bill gets {bill_votes} votes and passes')
    # else:
    #     print(f'The bundled bill gets {bill_votes} votes and fails')
    # print(f'The bundled bill had a total utilty of {bill_util}')
    
    
    
    '''Majority Rule'''
    passed_quanta_mr = 0
    vote_value_mr = 0
    for i in range(m):
        quanta_votes = 0
        total_util = 0
        for j in range(n):
            total_util += utilities[j][i]
            if utilities[j][i]>0:
                quanta_votes += 1
        if quanta_votes>n/2:
            passed_quanta_mr += 1
            vote_value_mr += total_util
        else:
            vote_value_mr -= total_util
    
    # print(f'Majority rule: {passed_quanta_mr} of {m} quanta passed')
    # print(f'Majority rule: Total value of majority rule vote is {vote_value_mr}')
    
    
    
    '''Utility Rule'''
    passed_quanta_ur = 0
    vote_value_ur = 0
        
    for i in range(m):
        quanta_votes = 0
        total_util = sum([utilities[j][i] for j in range(n)])
        
        if total_util > 0:
            passed_quanta_ur += 1
            vote_value_ur += total_util
            
        else:
            vote_value_ur -= total_util
    
    # print(f'Utility rule: {passed_quanta_ur} of {m} quanta passed')
    # print(f'Utility rule: Total value of utility rule vote is {vote_value_ur}')
    
    
    
    '''Vote Trading'''
    traded_utils = copy.deepcopy(utilities)
    stuck = False
    stuck_counter = 0
    indx = 0
    while stuck == False:
        potential_traders = list(range(indx+1, n))+list(range(indx))
        trade = False
        for nindx in potential_traders:
            #check if there is viable trade
            #q1 and q2 are positive trades
            #q3 and q4 are negative trades
            q1 = -1
            q2 = -1
            for quanta in range(m):
                if traded_utils[indx][quanta]==2 and traded_utils[nindx][quanta]==-1:
                    q1 = quanta
            for quanta in range(m):
                if traded_utils[indx][quanta]==-1 and traded_utils[nindx][quanta]==2:
                    q2 = quanta
            
            q3 = -1
            q4 = -1
            for quanta in range(m):
                if traded_utils[indx][quanta]==-2 and traded_utils[nindx][quanta]==1:
                    q3 = quanta
            for quanta in range(m):
                if traded_utils[indx][quanta]==1 and traded_utils[nindx][quanta]==-2:
                    q4 = quanta
            
            
            if (q1>-1 and q2>-1) or (q3>-1 and q4>-1):
                trade = True
                break
            
        if trade == True:
            #do the trade
            
            if q1>-1 and q2>-1 and q3>-1 and q4>-1:
                if rand.random()>0.5:
                    traded_utils[nindx][q1]=1.5
                    traded_utils[indx][q2]=1.5
                    stuck_counter = 0
                else:
                    traded_utils[nindx][q3]=-1.5
                    traded_utils[indx][q4]=-1.5
                    stuck_counter = 0
                    
            elif q1>-1 and q2>-1:
                traded_utils[nindx][q1]=1.5
                traded_utils[indx][q2]=1.5
                stuck_counter = 0
                
            else:
                traded_utils[nindx][q3]=-1.5
                traded_utils[indx][q4]=-1.5
                stuck_counter = 0   
    
    
            #print(f'{indx} and {nindx} traded votes on {q1} and {q2}')
        else:
            stuck_counter += 1
        if stuck_counter == n:
            stuck=True
        indx = (indx+1)%n
    
    passed_quanta_vt = 0
    vote_value_vt = 0
    for i in range(m):
        quanta_votes = 0
        total_util = 0
        for j in range(n):
            total_util += utilities[j][i]
            if traded_utils[j][i]>0:
                quanta_votes += 1
        if quanta_votes>n/2:
            passed_quanta_vt += 1
            vote_value_vt += total_util
        else:
            vote_value_vt -= total_util
    
    # print(f'Vote trading: {passed_quanta_vt} of {m} quanta passed')
    # print(f'Vote trading: Total value of vote trading vote is {vote_value_vt}')
    
    
    return [bill_votes, bill_util, passed_quanta_mr, vote_value_mr, passed_quanta_ur, vote_value_ur, passed_quanta_vt, vote_value_vt]
   

# voter_num=501
# quanta_num=51
# iters = 100

# p1 = 0.05
# p2 = 0.55
# p3 = 0.14
# p4 = 0.26

# probabilities = [p1, p2, p3, p4]

# bundle_values = []
# mr_values = []
# ur_values = []
# vt_values = []
# for _ in range(iters):
#     results = vote_sim(n=voter_num, m=quanta_num, method=2, probs=probabilities)
#     bill_votes = results[0] 
#     bill_util = results[1] 
#     passed_quanta_mr = results[2] 
#     vote_value_mr = results[3] 
#     passed_quanta_ur = results[4] 
#     vote_value_ur = results[5] 
#     passed_quanta_vt = results[6] 
#     vote_value_vt = results[7]
    
#     if bill_votes>voter_num/2:
#         bundle_values.append(bill_util)
#         # print('quantum passed')
#         # print(bundle_values[-1])
#     else:
#         bundle_values.append(-1*bill_util)
#         # print('quantum failed')
#         # print(bundle_values[-1])
        
#     mr_values.append(vote_value_mr)
#     ur_values.append(vote_value_ur)
#     vt_values.append(vote_value_vt)
    

# bundle_mean = np.mean(bundle_values)
# mr_mean = np.mean(mr_values)
# ur_mean = np.mean(ur_values)
# vt_mean = np.mean(vt_values)

# bundle_std = np.std(bundle_values)
# mr_std = np.std(mr_values)
# ur_std = np.std(ur_values)
# vt_std = np.std(vt_values)

# confidence = 0.95
# bundle_conf = st.norm.interval(alpha=confidence, loc=bundle_mean, scale=bundle_std/math.sqrt(iters))
# mr_conf = st.norm.interval(alpha=confidence, loc=mr_mean, scale=mr_std/math.sqrt(iters))
# ur_conf = st.norm.interval(alpha=confidence, loc=ur_mean, scale=ur_std/math.sqrt(iters))
# vt_conf = st.norm.interval(alpha=confidence, loc=vt_mean, scale=vt_std/math.sqrt(iters))

# voting_methods = ['Bundle', 'Majority Rule', 'Utility Rule', 'Vote Trading']
# x_vals = [0,2,4,6]
# mean_vals = [bundle_mean, mr_mean, ur_mean, vt_mean]
# std_vals = [bundle_std, mr_std, ur_std, vt_std]

# analytic_vals = []
# x_vals_2 = [1,3,5,7]

# # Bundle analytic approx
# if 2*p1 + p2 != p3 + 2*p4:
#     analytic_vals.append(np.abs(-2*p1-p2+p3+2*p4)*voter_num*quanta_num)
# else:
#     analytic_vals.append(4*np.sqrt(voter_num*quanta_num*(p1+p2)*(p3+p4))*(2*(p1+p4)+p2+p3)/math.pi)

# # MR analytic approx
# if p1+p2>p3+p4:
#     analytic_vals.append(voter_num*quanta_num*(2*p1+p2-p3-2*p4))
# elif p1+p2<p3+p4:
#     analytic_vals.append(voter_num*quanta_num*(-2*p1-p2+p3+2*p4))
# else:
#     analytic_vals.append(quanta_num*np.sqrt(2*voter_num/math.pi)*(2*(p1+p4)+p2+p3))
    
# # UR analytic approx
# if 2*p1+p2 != p3+2*p4:
#     analytic_vals.append(np.abs(-2*p1-p2+p3+2*p4)*voter_num*quanta_num)
# else:
#     analytic_vals.append(quanta_num*np.sqrt(8*voter_num/math.pi)*np.sqrt((2*p1+p2)*(p3+2*p4)))

# # VT analytic approx
# t1 = voter_num*quanta_num*(p2-np.sqrt(1-0.5**(1/(voter_num*(voter_num-1)*quanta_num*(quanta_num-1)/4))/2*p4*p4)/2)
# t2 = voter_num*quanta_num*(p3-np.sqrt(1-0.5**(1/(voter_num*(voter_num-1)*quanta_num*(quanta_num-1)/4))/2*p1*p1)/2)

# if p1+p2+2*(t2-t1)/(voter_num*quanta_num)>p4+p3+2*(t1-t2)/(voter_num*quanta_num):
#     analytic_vals.append(voter_num*quanta_num*(2*p1+p2-p3-2*p4))
# elif p1+p2+2*(t2-t1)/(voter_num*quanta_num)<p4+p3+2*(t1-t2)/(voter_num*quanta_num):
#     analytic_vals.append(voter_num*quanta_num*(2*p4+p3-p2-2*p1))
# else:
#     analytic_vals.append(quanta_num*np.sqrt(8*voter_num/math.pi)*np.sqrt((p2+p4)*(p1+p3))*((2*p4-p2)+(2*p1-p3)))

# analytic_vals = [0,0,0,0]

# fig, ax = plt.subplots()
# ax.bar(x_vals, mean_vals, yerr = [bundle_mean-bundle_conf[0], mr_mean-mr_conf[0], ur_mean-ur_conf[0], vt_mean-vt_conf[0]], align='center', alpha=0.5, ecolor='black', capsize=10)
# ax.bar(x_vals_2, analytic_vals, align='center', alpha=0.5)
# ax.set_xticks([0.5, 2.5, 4.5, 6.5])
# ax.set_xticklabels(voting_methods)
# ax.set_ylabel('Value from Voting')
# # ax.set_title(f'Vote values with probs {probabilities}')
# ax.set_title('Vote values with Spatial Model of Utility')






def vote_sim2(n, m, method):   
    utils = [-2,-1,1,2]
    
    '''Generate utilities for each of the voters and the utilities'''
    ##########
    # Method 1: Uniform and noisy
    ##########
    if method == 1:
        # probs = [0.25, 0.25, 0.25, 0.25]
        # probs = [0.2,0.35,0.2,0.25]
        # probs = [0,0.5,0.5,0]
        # probs = [0.3, 0.7, 0.7, 0.3]
        probs = [0.25, 0.25, 0.25, 0.25]
        utilities = [rand.choices(utils, weights=probs, k=m) for _ in range(n)]

    '''Bundled Voting'''
    # All quanta are placed in a bill and each voter supports the bill 
    # if it has a sum positive utility
    
    # bill_contents = [1 for _ in range(m)]
    # bill_votes = 0
    # bill_util = 0
    # for i in range(n):
    #     total_util = 0
    #     for j in range(m):
    #         total_util += bill_contents[j]*utilities[i][j]
    #     bill_util += total_util
    #     if total_util>0:
    #         bill_votes += 1
            

    '''Vote Trading'''
    traded_utils = copy.deepcopy(utilities)
    stuck = False
    stuck_counter = 0
    indx = 0
    pos_trades = 0
    neg_trades = 0
    while stuck == False:
        potential_traders = list(range(indx+1, n))+list(range(indx))
        trade = False
        for nindx in potential_traders:
            #check if there is viable trade
            #q1 and q2 are positive trades
            #q3 and q4 are negative trades
            q1 = -1
            q2 = -1
            for quanta in range(m):
                if traded_utils[indx][quanta]==2 and traded_utils[nindx][quanta]==-1:
                    q1 = quanta
            for quanta in range(m):
                if traded_utils[indx][quanta]==-1 and traded_utils[nindx][quanta]==2:
                    q2 = quanta
            
            q3 = -1
            q4 = -1
            for quanta in range(m):
                if traded_utils[indx][quanta]==-2 and traded_utils[nindx][quanta]==1:
                    q3 = quanta
            for quanta in range(m):
                if traded_utils[indx][quanta]==1 and traded_utils[nindx][quanta]==-2:
                    q4 = quanta
            
            
            if (q1>-1 and q2>-1) or (q3>-1 and q4>-1):
                trade = True
                break
            
        if trade == True:
            #do the trade
            
            if q1>-1 and q2>-1 and q3>-1 and q4>-1:
                if rand.random()>0.5:
                    traded_utils[nindx][q1]=1.5
                    traded_utils[indx][q2]=1.5
                    stuck_counter = 0
                    pos_trades += 1
                else:
                    traded_utils[nindx][q3]=-1.5
                    traded_utils[indx][q4]=-1.5
                    stuck_counter = 0
                    neg_trades += 1
                    
            elif q1>-1 and q2>-1:
                traded_utils[nindx][q1]=1.5
                traded_utils[indx][q2]=1.5
                stuck_counter = 0
                pos_trades += 1
                
            else:
                traded_utils[nindx][q3]=-1.5
                traded_utils[indx][q4]=-1.5
                stuck_counter = 0   
                neg_trades += 1
    
    
            #print(f'{indx} and {nindx} traded votes on {q1} and {q2}')
        else:
            stuck_counter += 1
        if stuck_counter == n:
            stuck=True
        indx = (indx+1)%n
    
    passed_quanta_vt = 0
    vote_value_vt = 0
    for i in range(m):
        quanta_votes = 0
        total_util = 0
        for j in range(n):
            total_util += utilities[j][i]
            if traded_utils[j][i]>0:
                quanta_votes += 1
        if quanta_votes>n/2:
            passed_quanta_vt += 1
            vote_value_vt += total_util
        else:
            vote_value_vt -= total_util
    
    # print(f'Vote trading: {passed_quanta_vt} of {m} quanta passed')
    # print(f'Vote trading: Total value of vote trading vote is {vote_value_vt}')

   
    return [vote_value_vt, passed_quanta_vt, pos_trades, neg_trades,0,0,0,0]

def matrix_flatten(matrix):
    n,m = np.shape(matrix)
    new_matrix = np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):
            if matrix[i][j]>0:
                new_matrix[i,j]=1
            else:
                new_matrix[i,j]=-1
    
    return new_matrix

def vote_trade_dimensions(n, m, method):   
    utils = [-2,-1,1,2]
    
    '''Generate utilities for each of the voters and the utilities'''
    ##########
    # Method 1: Uniform and noisy
    ##########
    if method == 1:
        # probs = [0.25, 0.25, 0.25, 0.25]
        # probs = [0.2,0.35,0.2,0.25]
        # probs = [0,0.5,0.5,0]
        # probs = [0.3, 0.7, 0.7, 0.3]
        probs = [0.25, 0.25, 0.25, 0.25]
        utilities = [rand.choices(utils, weights=probs, k=m) for _ in range(n)]
        
    ##########
    # Method 2: Spatial model of voting
    ##########
    elif method == 2:
        utilities = [[] for _ in range(n)]
        
        # dim = 2
        dim = 5
        voter_positions = [[rand.random() for _ in range(dim)] for _ in range(n)]
        for temp in range(m):
            sq_pos = np.asarray([rand.random() for _ in range(dim)])
            quantum_pos = np.asarray([rand.random() for _ in range(dim)])
            
            # Old method that uses perpendicular lines to line connecting sq and quantum
            for i in range(n):
                v_pos = np.asarray(voter_positions[i])
                if dist(v_pos, sq_pos) > dist(v_pos, quantum_pos):
                    temp = 2*quantum_pos - sq_pos
                    if dist(v_pos, temp)>dist(v_pos,sq_pos):
                        utilities[i].append(utils[2])
                    else:
                        utilities[i].append(utils[3])
                else:
                    temp = 2*sq_pos - quantum_pos
                    if dist(v_pos, temp)>dist(v_pos, quantum_pos):
                        utilities[i].append(utils[1])
                    else:
                        utilities[i].append(utils[0])
            
            # second method that has curved boundaries between utility domains
            # for i in range(n):
            #     v_pos = np.asarray(voter_positions[i])
            #     if dist(v_pos, sq_pos) > dist(v_pos, quantum_pos):
            #         if 3*dist(v_pos, sq_pos) > dist(v_pos, quantum_pos):
            #             utilities[i].append(utils[3])
            #         else:
            #             utilities[i].append(utils[2])
            #     else:
            #         if dist(v_pos, sq_pos) < 3*dist(v_pos, quantum_pos):
            #             utilities[i].append(utils[0])
            #         else:
            #             utilities[i].append(utils[1])
            

    '''Vote Trading'''
    traded_utils = copy.deepcopy(utilities)
    stuck = False
    stuck_counter = 0
    indx = 0
    pos_trades = 0
    neg_trades = 0
    while stuck == False:
        potential_traders = list(range(indx+1, n))+list(range(indx))
        trade = False
        for nindx in potential_traders:
            #check if there is viable trade
            #q1 and q2 are positive trades
            #q3 and q4 are negative trades
            q1 = -1
            q2 = -1
            for quanta in range(m):
                if traded_utils[indx][quanta]==2 and traded_utils[nindx][quanta]==-1:
                    q1 = quanta
            for quanta in range(m):
                if traded_utils[indx][quanta]==-1 and traded_utils[nindx][quanta]==2:
                    q2 = quanta
            
            q3 = -1
            q4 = -1
            for quanta in range(m):
                if traded_utils[indx][quanta]==-2 and traded_utils[nindx][quanta]==1:
                    q3 = quanta
            for quanta in range(m):
                if traded_utils[indx][quanta]==1 and traded_utils[nindx][quanta]==-2:
                    q4 = quanta
            
            
            if (q1>-1 and q2>-1) or (q3>-1 and q4>-1):
                trade = True
                break
            
        if trade == True:
            #do the trade
            
            if q1>-1 and q2>-1 and q3>-1 and q4>-1:
                if rand.random()>0.5:
                    traded_utils[nindx][q1]=1.5
                    traded_utils[indx][q2]=1.5
                    stuck_counter = 0
                    pos_trades += 1
                else:
                    traded_utils[nindx][q3]=-1.5
                    traded_utils[indx][q4]=-1.5
                    stuck_counter = 0
                    neg_trades += 1
                    
            elif q1>-1 and q2>-1:
                traded_utils[nindx][q1]=1.5
                traded_utils[indx][q2]=1.5
                stuck_counter = 0
                pos_trades += 1
                
            else:
                traded_utils[nindx][q3]=-1.5
                traded_utils[indx][q4]=-1.5
                stuck_counter = 0   
                neg_trades += 1
    
    
            #print(f'{indx} and {nindx} traded votes on {q1} and {q2}')
        else:
            stuck_counter += 1
        if stuck_counter == n:
            stuck=True
        indx = (indx+1)%n
    
    passed_quanta_vt = 0
    vote_value_vt = 0
    for i in range(m):
        quanta_votes = 0
        total_util = 0
        for j in range(n):
            total_util += utilities[j][i]
            if traded_utils[j][i]>0:
                quanta_votes += 1
        if quanta_votes>n/2:
            passed_quanta_vt += 1
            vote_value_vt += total_util
        else:
            vote_value_vt -= total_util
    
    # print(f'Vote trading: {passed_quanta_vt} of {m} quanta passed')
    # print(f'Vote trading: Total value of vote trading vote is {vote_value_vt}')

    pca_untraded=PCA(n_components = n)
    pca_untraded.fit(matrix_flatten(utilities))
    # pca_untraded.fit(utilities)
    
    pca_traded=PCA(n_components = n)
    pca_traded.fit(matrix_flatten(traded_utils))
    # pca_traded.fit(traded_utils)
   
    return [vote_value_vt, passed_quanta_vt, pca_untraded.explained_variance_, pca_traded.explained_variance_,pos_trades,neg_trades,0]

def vote_trade_dims_coalition(n, m, method):   
    utils = [-2,-1,1,2]
    
    '''Generate utilities for each of the voters and the utilities'''
    ##########
    # Method 1: Uniform and noisy
    ##########
    if method == 1:
        # probs = [0.25, 0.25, 0.25, 0.25]
        # probs = [0.2,0.35,0.2,0.25]
        # probs = [0,0.5,0.5,0]
        # probs = [0.3, 0.7, 0.7, 0.3]
        probs = [0.25, 0.25, 0.25, 0.25]
        utilities = [rand.choices(utils, weights=probs, k=m) for _ in range(n)]
        voter_coalition = [rand.random() for _ in range(n)]
        
    ##########
    # Method 2: Spatial model of voting
    ##########
    elif method == 2:
        utilities = [[] for _ in range(n)]
        
        # dim = 2
        dim = 5
        voter_positions = [[rand.random() for _ in range(dim)] for _ in range(n)]
        voter_coalition = [rand.random() for _ in range(n)]
        for temp in range(m):
            sq_pos = np.asarray([rand.random() for _ in range(dim)])
            quantum_pos = np.asarray([rand.random() for _ in range(dim)])
            
            # Old method that uses perpendicular lines to line connecting sq and quantum
            for i in range(n):
                v_pos = np.asarray(voter_positions[i])
                if dist(v_pos, sq_pos) > dist(v_pos, quantum_pos):
                    temp = 2*quantum_pos - sq_pos
                    if dist(v_pos, temp)>dist(v_pos,sq_pos):
                        utilities[i].append(utils[2])
                    else:
                        utilities[i].append(utils[3])
                else:
                    temp = 2*sq_pos - quantum_pos
                    if dist(v_pos, temp)>dist(v_pos, quantum_pos):
                        utilities[i].append(utils[1])
                    else:
                        utilities[i].append(utils[0])
            
            # second method that has curved boundaries between utility domains
            # for i in range(n):
            #     v_pos = np.asarray(voter_positions[i])
            #     if dist(v_pos, sq_pos) > dist(v_pos, quantum_pos):
            #         if 3*dist(v_pos, sq_pos) > dist(v_pos, quantum_pos):
            #             utilities[i].append(utils[3])
            #         else:
            #             utilities[i].append(utils[2])
            #     else:
            #         if dist(v_pos, sq_pos) < 3*dist(v_pos, quantum_pos):
            #             utilities[i].append(utils[0])
            #         else:
            #             utilities[i].append(utils[1])
            

    '''Vote Trading'''
    traded_utils = copy.deepcopy(utilities)
    stuck = False
    stuck_counter = 0
    indx = 0
    pos_trades = 0
    neg_trades = 0
    while stuck == False:
        potential_traders = list(range(indx+1, n))+list(range(indx))
        trade = False
        for nindx in potential_traders:
            #check if there is viable trade
            #q1 and q2 are positive trades
            #q3 and q4 are negative trades
            q1=-1
            q2=-1
            q3=-1
            q4=-1
            #only if they are in the same coalition
            # if vote_coalition[indx]==voter_coalition[nindx]:
            # if np.abs(voter_positions[indx][0]-voter_positions[nindx][0])<0.1 or np.abs(voter_positions[indx][1]-voter_positions[nindx][1])<0.1:
            # if np.abs(voter_positions[indx][0]-voter_positions[nindx][0])<0.2: 
            if np.abs(voter_coalition[indx]-voter_coalition[nindx])<0.02: 
                for quanta in range(m):
                    if traded_utils[indx][quanta]==2 and traded_utils[nindx][quanta]==-1:
                        q1 = quanta
                for quanta in range(m):
                    if traded_utils[indx][quanta]==-1 and traded_utils[nindx][quanta]==2:
                        q2 = quanta
                
                for quanta in range(m):
                    if traded_utils[indx][quanta]==-2 and traded_utils[nindx][quanta]==1:
                        q3 = quanta
                for quanta in range(m):
                    if traded_utils[indx][quanta]==1 and traded_utils[nindx][quanta]==-2:
                        q4 = quanta
            
            
            if (q1>-1 and q2>-1) or (q3>-1 and q4>-1):
                trade = True
                break
            
        if trade == True:
            #do the trade
            
            if q1>-1 and q2>-1 and q3>-1 and q4>-1:
                if rand.random()>0.5:
                    traded_utils[nindx][q1]=1.5
                    traded_utils[indx][q2]=1.5
                    stuck_counter = 0
                    pos_trades += 1
                else:
                    traded_utils[nindx][q3]=-1.5
                    traded_utils[indx][q4]=-1.5
                    stuck_counter = 0
                    neg_trades += 1
                    
            elif q1>-1 and q2>-1:
                traded_utils[nindx][q1]=1.5
                traded_utils[indx][q2]=1.5
                stuck_counter = 0
                pos_trades += 1
                
            else:
                traded_utils[nindx][q3]=-1.5
                traded_utils[indx][q4]=-1.5
                stuck_counter = 0   
                neg_trades += 1
    
    
            #print(f'{indx} and {nindx} traded votes on {q1} and {q2}')
        else:
            stuck_counter += 1
        if stuck_counter == n:
            stuck=True
        indx = (indx+1)%n
    
    passed_quanta_vt = 0
    vote_value_vt = 0
    for i in range(m):
        quanta_votes = 0
        total_util = 0
        for j in range(n):
            total_util += utilities[j][i]
            if traded_utils[j][i]>0:
                quanta_votes += 1
        if quanta_votes>n/2:
            passed_quanta_vt += 1
            vote_value_vt += total_util
        else:
            vote_value_vt -= total_util
    
    # print(f'Vote trading: {passed_quanta_vt} of {m} quanta passed')
    # print(f'Vote trading: Total value of vote trading vote is {vote_value_vt}')

    pca_untraded=PCA(n_components = n)
    pca_untraded.fit(matrix_flatten(utilities))
    # pca_untraded.fit(utilities)
    
    pca_traded=PCA(n_components = n)
    pca_traded.fit(matrix_flatten(traded_utils))
    # pca_traded.fit(traded_utils)
   
    return [vote_value_vt, passed_quanta_vt, pca_untraded.explained_variance_, pca_traded.explained_variance_,pos_trades,neg_trades,0]


voter_num = 101
quanta_num = 101
iters = 100

vt_value_list = []
vt_num_passed = []
var_untraded_list = []
var_traded_list = []
pos_trades_list = []
neg_trades_list = []

for _ in range(iters):
    results = vote_trade_dims_coalition(n=voter_num, m=quanta_num, method=1)
    vote_value_vt = results[0] 
    passed_quanta_vt = results[1] 
    untraded_var = results[2] 
    traded_var = results[3] 
    
    vt_value_list.append(vote_value_vt)
    vt_num_passed.append(passed_quanta_vt)
    var_untraded_list.append(untraded_var)
    var_traded_list.append(traded_var)
    pos_trades_list.append(results[4])
    neg_trades_list.append(results[5])
    
print(np.mean(vt_value_list))
print(np.mean(vt_num_passed))
print(np.mean(pos_trades_list))
print(np.mean(neg_trades_list))

plt.subplots()
plt.plot(list(range(voter_num)), np.mean(var_untraded_list, axis=0))
plt.plot(list(range(voter_num)), np.mean(var_traded_list, axis=0))
plt.ylabel('Explained Variance')
plt.xlabel('Principal Component')

interval = 20
plt.subplots()
plt.plot(list(range(interval)), np.mean(var_untraded_list, axis=0)[0:interval])
plt.plot(list(range(interval)), np.mean(var_traded_list, axis=0)[0:interval])
plt.ylabel('Explained Variance')
plt.xlabel('Principal Component')

