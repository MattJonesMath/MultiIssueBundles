###########################################
# Cleaner code for vote trading model
###########################################

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


def vote_trade(n, m, util_method, analyses, utils = [-2,-1,1,2], util_probs = [0.25,0.25,0.25,0.25], spatial_dim=2, trade_net='complete', trade_thresh=1):
    
    results = {}
    
    ####################
    # generate utilities
    ####################
    if util_method == 'iid':
        utilities = [rand.choices(utils, weights=util_probs, k=m) for _ in range(n)]
        
        
    elif util_method == 'spatial':
        utilities = [[] for _ in range(n)]
    
        voter_positions = [[rand.random() for _ in range(spatial_dim)] for _ in range(n)]
        for _ in range(m):
            no_pos = np.asarray([rand.random() for _ in range(spatial_dim)])
            yes_pos = np.asarray([rand.random() for _ in range(spatial_dim)])
            # Use perpendicular lines to line connecting sq and quantum
            for i in range(n):
                v_pos = np.asarray(voter_positions[i])
                if dist(v_pos, no_pos) > dist(v_pos, yes_pos):
                    temp = 2*yes_pos - no_pos
                    if dist(v_pos, temp)>dist(v_pos,no_pos):
                        utilities[i].append(utils[2])
                    else:
                        utilities[i].append(utils[3])
                else:
                    temp = 2*no_pos - yes_pos
                    if dist(v_pos, temp)>dist(v_pos, yes_pos):
                        utilities[i].append(utils[1])
                    else:
                        utilities[i].append(utils[0])
                        
                        
    else:
        print('error')
        return 
    
    results['utilities'] = utilities
    
    
    ######################
    # Conduct analyses
    ######################
            
            
    if 'maj_rule' in analyses:
        issues_passed_mr = 0
        vote_value_mr = 0
        maj_support = []
        for i in range(m):
            ind_votes = 0
            total_util = 0
            for j in range(n):
                total_util += utilities[j][i]
                if utilities[j][i]>0:
                    ind_votes += 1
            if ind_votes>n/2:
                issues_passed_mr += 1
                vote_value_mr += total_util
                maj_support.append(1)
            else:
                vote_value_mr -= total_util
                maj_support.append(-1)
                
        results['mr_issues_passed'] = issues_passed_mr
        results['mr_val'] = vote_value_mr
    
    
    
    if 'util_rule' in analyses:
        issues_passed_ur = 0
        vote_value_ur = 0
        for i in range(m):
            total_util = sum([utilities[j][i] for j in range(n)])
            if total_util > 0:
                issues_passed_ur += 1
                vote_value_ur += total_util
            else:
                vote_value_ur -= total_util
                
        results['ur_issues_passed'] = issues_passed_ur
        results['ur_val'] = vote_value_ur
        
        
        
    if 'bundle' in analyses:
        bill_votes = 0
        bill_util = 0
        for i in range(n):
            total_util = 0
            for j in range(m):
                total_util += utilities[i][j]
            bill_util += total_util
            if total_util>0:
                bill_votes += 1
                
        results['bundle_votes']=bill_votes
        
        if bill_votes<n/2:
            results['bundle_val']=-1*bill_util
        else:
            results['bundle_val']=bill_util
    
    
    
    if 'vote_trade' in analyses or 'dimension' in analyses:
        votes_owned = [[[i] for _ in range(m)] for i in range(n)]
        trade_network = []
        
        if trade_net=='complete':
            for i in range(n):
                trade_network.append(list(range(i+1, n))+list(range(i)))
        else:
            print('No Network!')
        
        stuck = False
        stuck_counter = 0
        indx = 0
        trades = 0
        
        while not stuck:
            trade=False
            trade_pairs = []
            for i in range(1,m):
                for j in range(i):
                    if 0<len(votes_owned[indx][i])<n/2 and 0<len(votes_owned[indx][j])<n/2:
                        if np.abs(utilities[indx][i])-np.abs(utilities[indx][j]) >= trade_thresh:
                            trade_pairs.append([i,j])
                        elif np.abs(utilities[indx][j])-np.abs(utilities[indx][i]) >= trade_thresh:
                            trade_pairs.append([j,i])
            for nindx in trade_network[indx]:
                potential_trades = []
                for pair in trade_pairs:
                    i = pair[0]
                    j = pair[1]
                    if 0<len(votes_owned[nindx][i])<n/2 and 0<len(votes_owned[nindx][j])<n/2:
                        if np.abs(utilities[nindx][j])-np.abs(utilities[nindx][i])>=trade_thresh:
                            potential_trades.append(pair)
                if potential_trades:
                    trade=True
                    break
            
            if trade==True:
                trade_pair = rand.choice(potential_trades)
                i = trade_pair[0]
                j = trade_pair[1]
                if votes_owned[indx][j]==[indx]:
                    indx_give = indx
                else:
                    indx_give = rand.choice([k for k in votes_owned[indx][j] if k!=indx])
                if votes_owned[nindx][i]==[nindx]:
                    nindx_give = nindx
                else:
                    nindx_give = rand.choice([i for i in votes_owned[nindx][i]])
                
                votes_owned[indx][i].append(nindx_give)
                votes_owned[indx][j].remove(indx_give)
                votes_owned[nindx][j].append(indx_give)
                votes_owned[nindx][i].remove(nindx_give)
                
                stuck_counter=0
                trades+=1
                
            else:
                stuck_counter += 1
                
            if stuck_counter == n:
                stuck=True
                
            indx = (indx+1)%n
            
        results['total_trade_num'] = trades
        results['votes_owned'] = votes_owned
        
        traded_votes = []
        for indx in range(n):
            indx_votes = []
            for t in range(m):
                nindx = [i for i in range(n) if indx in votes_owned[i][t]][0]
                if utilities[nindx][t]>0:
                    indx_votes.append(1)
                else:
                    indx_votes.append(-1)
            traded_votes.append(indx_votes)
        
        results['traded_votes'] = traded_votes
    
    
    return results
        
    
x=vote_trade(10,5,'iid',['vote_trade'])
    
    
    
    
    
    
    
    
    