###############################################################
# Bundled multi-issue decision making
###############################################################
# examine the effect of making multiple decisions
# in a single vote.
###############################################################

import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import random as rand
import json
import copy
import scipy.stats as st
import math
from sklearn.decomposition import PCA
from voters_class import Voters
from sklearn.linear_model import LinearRegression



SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



############################################
# Create plots of vi pair and maj-sup scores
############################################

# voter_num = 101
# issue_num = 101
# null_val = voter_num*issue_num*0.5
# iters=10000

# vi_pair_scores = []
# maj_sup_scores = []

# for _ in range(iters):
#     voters = Voters(n=voter_num,m=issue_num,util_method='iid',util_probs=[0,0.5,0.5,0])
#     vi_pair_scores.append((2*voters.bundle_vi_pair_agreement-(voter_num*issue_num))/(voter_num*issue_num))
#     maj_sup_scores.append((2*voters.bundle_maj_agreement-issue_num)/issue_num)

# plt.subplots()
# plt.hist(vi_pair_scores)
# analytic_mean = (2/math.pi*math.sqrt(voter_num*issue_num))/(voter_num*issue_num)
# sim_mean_1 = np.mean(vi_pair_scores)
# plt.plot([sim_mean_1,sim_mean_1],[0,3500])
# # plt.legend(['Simulation mean'])
# plt.plot([analytic_mean,analytic_mean],[0,3500],'--')
# plt.legend(['Simulation mean','Analytic mean'])
# plt.xlabel('Utility Score')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('fig1a.png', dpi=300)


# plt.subplots()
# plt.hist(maj_sup_scores)
# sim_mean_2 = np.mean(maj_sup_scores)
# plt.plot([sim_mean_2,sim_mean_2],[0,3500])
# plt.legend(['Simulation mean'])
# plt.xlabel('Issue Score')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('fig1b.png', dpi=300)



# vi_pair_scores = []
# maj_sup_scores = []

# for _ in range(iters):
#     voters = Voters(n=voter_num,m=issue_num,util_method='spatial',util_probs=[0,0.5,0.5,0])
#     vi_pair_scores.append((2*voters.bundle_vi_pair_agreement-(voter_num*issue_num))/(voter_num*issue_num))
#     maj_sup_scores.append((2*voters.bundle_maj_agreement-issue_num)/issue_num)

# plt.subplots()
# plt.hist(vi_pair_scores)
# sim_mean_1 = np.mean(vi_pair_scores)
# plt.plot([sim_mean_1,sim_mean_1],[0,3000])
# plt.legend(['Simulation mean'])
# plt.xlabel('Utility Score')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('fig1c.png', dpi=300)



# plt.subplots()
# plt.hist(maj_sup_scores)
# sim_mean_2 = np.mean(maj_sup_scores)
# plt.plot([sim_mean_2,sim_mean_2],[0,4000])
# plt.legend(['Simulation mean'])
# plt.xlabel('Issue Score')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('fig1d.png', dpi=300)




#########################################
# Create plots comparing value of bundling and maj rule 
# as iid variables become less 50-50
#########################################

# voter_num = 101
# issue_num = 51
# null_val = voter_num*issue_num*0.5

# ps = [0.5+i/100 for i in range(16)]

# bundle_vi_pair_agree_list = []
# maj_vi_pair_agree_list = []
# bundle_maj_issue_agree_list = []

# iters = 1000

# for p in ps:
#     print(p)
#     bundle_vi_pair_agreements = []
#     maj_vi_pair_agreements = []
#     bundle_maj_issue_agreements = []
    
#     for _ in range(iters):
#         voters = Voters(n=voter_num, m=issue_num, util_method='iid', util_probs=[0,1-p,p,0])
#         bundle_vi_pair_agreements.append(voters.bundle_vi_pair_agreement)
#         maj_vi_pair_agreements.append(voters.maj_vi_pair_agreement)
#         bundle_maj_issue_agreements.append(voters.bundle_maj_agreement)
        
#     bundle_vi_pair_agree_list.append(np.mean(bundle_vi_pair_agreements))
#     maj_vi_pair_agree_list.append(np.mean(maj_vi_pair_agreements))
#     bundle_maj_issue_agree_list.append(np.mean(bundle_maj_issue_agreements))
    

# plt.subplots()
# plt.plot(ps, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in bundle_vi_pair_agree_list])
# plt.plot(ps, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in maj_vi_pair_agree_list])
# plt.plot(ps, [2*x-1 for x in ps], '--')
# plt.xlabel('$p$')
# plt.ylabel('Expected Utility Score')
# plt.legend(['Bundled Vote','Issue-by-Issue Vote', 'Issue-by-Issue Analytic'])
# plt.tight_layout()
# plt.savefig('fig2a.png', dpi=300)

# plt.subplots()
# plt.plot(ps, [(2*x-issue_num)/issue_num for x in bundle_maj_issue_agree_list])
# plt.plot([0.5,ps[-1]], [1, 1])
# plt.xlabel('$p$')
# plt.ylabel('Expected Issue Score')
# plt.legend(['Bundled Vote', 'Issue-by-Issue Vote'])
# plt.tight_layout()
# plt.savefig('fig2b.png', dpi=300)


#########################################
# Investigating the value of bundle as voters and issues become
# skewed along the ideological dimensions
#########################################
# voter_num = 101
# issue_num = 51
# iters = 1000

# bundle_vi_pair_agree_list = []
# maj_vi_pair_agree_list = []
# bundle_maj_issue_agree_list = []

# qs = [0.5+i/30 for i in range(16)]

# for q in qs:
#     print(q)
#     bundle_vi_pair_agreements = []
#     maj_vi_pair_agreements = []
#     bundle_maj_issue_agreements = []
    
#     for _ in range(iters):
#         voters = Voters(n=voter_num, m=issue_num, util_method='spatial', spatial_skew=q)
#         bundle_vi_pair_agreements.append(voters.bundle_vi_pair_agreement)
#         maj_vi_pair_agreements.append(voters.maj_vi_pair_agreement)
#         bundle_maj_issue_agreements.append(voters.bundle_maj_agreement)
        
#     bundle_vi_pair_agree_list.append(np.mean(bundle_vi_pair_agreements))
#     maj_vi_pair_agree_list.append(np.mean(maj_vi_pair_agreements))
#     bundle_maj_issue_agree_list.append(np.mean(bundle_maj_issue_agreements))

# plt.subplots()
# plt.plot(qs, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in bundle_vi_pair_agree_list])
# plt.plot(qs, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in maj_vi_pair_agree_list])
# plt.xlabel('$q$')
# plt.ylabel('Expected Utility Score')
# plt.legend(['Bundled Vote','Issue-by-Issue Vote'])
# plt.tight_layout()
# plt.savefig('fig2c.png', dpi=300)

# plt.subplots()
# plt.plot(qs, [(2*x-issue_num)/issue_num for x in bundle_maj_issue_agree_list])
# plt.plot([qs[0],qs[-1]], [1, 1])
# plt.xlabel('$q$')
# plt.ylabel('Expected Issue Score')
# plt.legend(['Bundled Vote', 'Issue-by-Issue Vote'])
# plt.tight_layout()
# plt.savefig('fig2d.png', dpi=300)



########################################################
# Investigate the effect of breaking the bundle into smaller bundles
########################################################
# voter_num = 101
# issue_num = 101
# iters = 1000

# bundle_vi_pair_agree_list = []
# maj_vi_pair_agree_list = []
# bundle_maj_issue_agree_list = []

# # bundle_nums = [2*i+1 for i in range(51)]
# bundle_nums = list(range(1,issue_num))

# for num in bundle_nums:
#     if num%10==0:
#         print(num)
#     bundle_vi_pair_agreements = []
#     maj_vi_pair_agreements = []
#     bundle_maj_issue_agreements = []
    
#     for _ in range(iters):
#         voters = Voters(n=voter_num, m=issue_num, util_method='iid', bundle_split='random', bundle_num=num)
#         voters.maj_rule_vote()
#         voters.bundle_vote()
#         voters.split_bundle()
#         bundle_vi_pair_agreements.append(voters.split_bundle_vi_pair_agreement)
#         maj_vi_pair_agreements.append(voters.maj_vi_pair_agreement)
#         bundle_maj_issue_agreements.append(voters.split_bundle_maj_agreement)
        
#     bundle_vi_pair_agree_list.append(np.mean(bundle_vi_pair_agreements))
#     maj_vi_pair_agree_list.append(np.mean(maj_vi_pair_agreements))
#     bundle_maj_issue_agree_list.append(np.mean(bundle_maj_issue_agreements))

# plt.subplots()
# plt.plot(bundle_nums, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in bundle_vi_pair_agree_list])
# plt.plot(bundle_nums, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in maj_vi_pair_agree_list])
# plt.plot(bundle_nums, [2/(np.pi*np.sqrt(voter_num*issue_num))*np.sqrt(i) for i in bundle_nums])
# plt.xlabel('Number of Bundles')
# plt.ylabel('Expected Utility Score')
# plt.legend(['Split Bundle Vote','Issue-by-Issue Vote','Analytic Approximation'])
# plt.tight_layout()
# plt.savefig('fig3a.png', dpi=300)

# # plt.subplots()
# # plt.plot(bundle_nums, [(2*x-issue_num)/issue_num for x in bundle_maj_issue_agree_list])
# # plt.plot([bundle_nums[0],bundle_nums[-1]], [1, 1])
# # plt.xlabel('Number of Bundles')
# # plt.ylabel('Expected Issue Score')
# # plt.legend(['Bundled Vote', 'Issue-by-issue Majority Rule'])
# # plt.tight_layout()



###########################################
# Investigate breaking into odd-only, balanced size bundles
###########################################
# voter_num = 101
# issue_num = 101
# iters = 1000

# bundle_vi_pair_agree_list = []
# maj_vi_pair_agree_list = []
# bundle_maj_issue_agree_list = []

# bundle_nums = [2*i+1 for i in range(51)]
# # bundle_nums = list(range(1,issue_num))

# for num in bundle_nums:
#     if num%10==1:
#         print(num)
#     bundle_vi_pair_agreements = []
#     maj_vi_pair_agreements = []
#     bundle_maj_issue_agreements = []
    
#     for _ in range(iters):
#         voters = Voters(n=voter_num, m=issue_num, util_method='iid', bundle_split='balanced_random', bundle_num=num)
#         voters.maj_rule_vote()
#         voters.bundle_vote()
#         voters.split_bundle()
#         bundle_vi_pair_agreements.append(voters.split_bundle_vi_pair_agreement)
#         maj_vi_pair_agreements.append(voters.maj_vi_pair_agreement)
#         bundle_maj_issue_agreements.append(voters.split_bundle_maj_agreement)
        
#     bundle_vi_pair_agree_list.append(np.mean(bundle_vi_pair_agreements))
#     maj_vi_pair_agree_list.append(np.mean(maj_vi_pair_agreements))
#     bundle_maj_issue_agree_list.append(np.mean(bundle_maj_issue_agreements))

# plt.subplots()
# plt.plot(bundle_nums, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in bundle_vi_pair_agree_list])
# plt.plot(bundle_nums, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in maj_vi_pair_agree_list])
# plt.plot(bundle_nums, [2/(np.pi*np.sqrt(voter_num*issue_num))*np.sqrt(i) for i in bundle_nums])
# plt.xlabel('Number of Bundles')
# plt.ylabel('Expected Utility Score')
# plt.legend(['Split Bundle Vote','Issue-by-Issue Vote','Analytic Approximation'])
# plt.tight_layout()
# plt.savefig('fig3b.png', dpi=300)

# # plt.subplots()
# # plt.plot(bundle_nums, [(2*x-issue_num)/issue_num for x in bundle_maj_issue_agree_list])
# # plt.plot([bundle_nums[0],bundle_nums[-1]], [1, 1])
# # plt.xlabel('Number of Bundles')
# # plt.ylabel('Expected Issue Score')
# # plt.legend(['Bundled Vote', 'Issue-by-issue Majority Rule'])
# # plt.tight_layout()




###############################################################
# Investigate dividing the bundle into two bundles
# with IID model
###############################################################
# voter_num = 101
# issue_num = 101
# iters = 10000

# split_count = 0
# splitter_gains = []
# nonsplitter_gains = []
# bundle_maj = []
# split_maj = []
# splitter_counts = []
# nonsplit_pass_counts = []
# nonsplit_fail_counts = []
# bundle_pass_count = 0

# bundle_vi_pair_agree_list = []
# split_bundle_vi_pair_agree_list = []
# maj_vi_pair_agree_list = []

# vi_pair_scores = []
# gerrymander_count = 0

# for iter in range(iters):
#     if iter%1000==0:
#         print(iter)
#     voters = Voters(voter_num, issue_num, 'iid', util_probs=[0,0.5,0.5,0], bundle_split='random')
#     if voters.splitter_gain != 0:
#         split_count += 1
#     if voters.bundle_pass and voters.bundle_passage==[-1,-1]:
#         gerrymander_count += 1
#     if not voters.bundle_pass and voters.bundle_passage==[1,1]:
#         gerrymander_count += 1

#     splitter_gains.append(voters.splitter_gain)
#     nonsplitter_gains.append(voters.nonsplitter_gain)
#     bundle_maj.append(voters.bundle_maj_agreement)
#     split_maj.append(voters.split_bundle_maj_agreement)
#     splitter_counts.append(voters.splitter_count)
#     nonsplit_pass_counts.append(voters.nonsplit_pass)
#     nonsplit_fail_counts.append(voters.nonsplit_fail)
#     bundle_vi_pair_agree_list.append(voters.bundle_vi_pair_agreement)
#     split_bundle_vi_pair_agree_list.append(voters.split_bundle_vi_pair_agreement)
#     maj_vi_pair_agree_list.append(voters.maj_vi_pair_agreement)
    
#     vi_pair_scores.append((2*voters.split_bundle_vi_pair_agreement - voter_num*issue_num)/(voter_num*issue_num))
    
#     if voters.bundle_pass:
#         bundle_pass_count += 1
    

# print(f'Probability of a bundle split mattering is {split_count/iters}')
# print(f'Probability that both subbundles flip from bundle is {gerrymander_count/iters}')
# print(f'Expected gain for a splitter is {np.mean(splitter_gains)}')
# print(f'Expected gain for a non-splitter is {np.mean(nonsplitter_gains)}')
# print(f'Expected agreement between single bundle and majority rule is {np.mean(bundle_maj)}')
# print(f'Expected agreement between two bundles and majority rule is {np.mean(split_maj)}')

# print(f'Expected number of splitters is {np.mean(splitter_counts)}')
# print(f'Expected number of non-splitters that pass both subbundles is {np.mean(nonsplit_pass_counts)}')
# print(f'Expected number of non-splitters that fail both subbundles is {np.mean(nonsplit_fail_counts)}')
# print(f'Probability of a full bundle passing is {bundle_pass_count/iters}')

# print(f'Expected number of voter-issue pairs satisfied by majority rule is {np.mean(maj_vi_pair_agree_list)}')
# print(f'Expected number of voter-issue pairs satisfied by a single bundle is {np.mean(bundle_vi_pair_agree_list)}')
# print(f'Expected number of voter-issue pairs satisfied by a two bundles is {np.mean(split_bundle_vi_pair_agree_list)}')

# x_vals = [x/voter_num for x in splitter_counts] 
# y_vals = [(2*x - voter_num*issue_num)/(voter_num*issue_num) for x in bundle_vi_pair_agree_list]
# min_count = min(x_vals)
# max_count = max(x_vals)

# # reg = LinearRegression().fit(np.asarray(splitter_counts).reshape(-1,1), split_bundle_vi_pair_agree_list)
# reg = LinearRegression().fit(np.asarray(x_vals).reshape(-1,1), y_vals)
# intercept = reg.intercept_
# slope = reg.coef_[0]

# plt.subplots()
# plt.scatter(x_vals, y_vals, alpha=0.08)
# plt.plot([min_count, max_count],[intercept+slope*min_count, intercept+slope*max_count], c='r')
# plt.xlabel('Fraction of voters in favor of splitting')
# plt.ylabel('Single Bundle Utility Score')
# plt.tight_layout()
# plt.savefig('fig4a.png', dpi=300)


# x_vals = [x/voter_num for x in splitter_counts]
# y_vals = [(2*x - voter_num*issue_num)/(voter_num*issue_num) for x in split_bundle_vi_pair_agree_list]
# min_count = min(x_vals)
# max_count = max(x_vals)

# # reg = LinearRegression().fit(np.asarray(splitter_counts).reshape(-1,1), split_bundle_vi_pair_agree_list)
# reg = LinearRegression().fit(np.asarray(x_vals).reshape(-1,1), y_vals)
# intercept = reg.intercept_
# slope = reg.coef_[0]

# plt.subplots()
# plt.scatter(x_vals, y_vals, alpha=0.08)
# plt.plot([min_count, max_count],[intercept+slope*min_count, intercept+slope*max_count], c='r')
# plt.xlabel('Fraction of voters in favor of splitting')
# plt.ylabel('Subbundles Utility Score')
# plt.tight_layout()
# plt.savefig('fig4b.png', dpi=300)


# x_vals = [x/voter_num for x in splitter_counts]
# y_vals = [(2*split_bundle_vi_pair_agree_list[i]-voter_num*issue_num)/(voter_num*issue_num) - (2*bundle_vi_pair_agree_list[i]-voter_num*issue_num)/(voter_num*issue_num) for i in range(iters)]
# min_count = min(x_vals)
# max_count = max(x_vals)

# # reg = LinearRegression().fit(np.asarray(splitter_counts).reshape(-1,1), split_bundle_vi_pair_agree_list)
# reg = LinearRegression().fit(np.asarray(x_vals).reshape(-1,1), y_vals)
# intercept = reg.intercept_
# slope = reg.coef_[0]

# plt.subplots()
# plt.scatter(x_vals, y_vals, alpha=0.08)
# plt.plot([min_count, max_count],[intercept+slope*min_count, intercept+slope*max_count], c='r')
# plt.xlabel('Fraction of voters in favor of splitting')
# plt.ylabel('Utility Score Change from Split')
# plt.tight_layout()
# plt.savefig('fig4c.png', dpi=300)



########################################
# Create table of frequency of
# subbundle flipping entire vote
########################################

voter_nums = [11, 101, 1001]
issue_nums = [11, 101, 1001]

for voter_num in voter_nums:
    for issue_num in issue_nums:
        iters = 100
        samples = 100

        iter_list = []
        for iter in range(iters):
            sample_list = []
            for sample in range(samples):
                gerrymander_count = 0
                # if iter%1000==0:
                #     print(iter)
                voters = Voters(voter_num, issue_num, 'iid', util_probs=[0,0.5,0.5,0], bundle_split='random')
                if voters.bundle_pass and voters.bundle_passage==[-1,-1]:
                    gerrymander_count += 1
                if not voters.bundle_pass and voters.bundle_passage==[1,1]:
                    gerrymander_count += 1
                sample_list.append(gerrymander_count)
            iter_list.append(st.tmean(sample_list))
        
        prob = st.tmean(iter_list)
        sd = st.tstd(iter_list)
                
        print(f'n: {voter_num}, m: {issue_num}')
        print(f'Probability that both subbundles flip from bundle is {prob} pm {sd}')



###############################################################
# Investigate dividing the bundle into two bundles
# with spatial model
###############################################################
# voter_num = 101
# issue_num = 101
# iters = 10000

# split_count = 0
# splitter_gains = []
# nonsplitter_gains = []
# bundle_maj = []
# split_maj = []
# splitter_counts = []
# nonsplit_pass_counts = []
# nonsplit_fail_counts = []
# bundle_pass_count = 0

# bundle_vi_pair_agree_list = []
# split_bundle_vi_pair_agree_list = []
# maj_vi_pair_agree_list = []

# vi_pair_scores = []
# gerrymander_count = 0

# for iter in range(iters):
#     if iter%500==0:
#         print(iter)
#     voters = Voters(voter_num, issue_num, 'spatial', util_probs=[0,0.5,0.5,0], bundle_split='random')
#     if voters.splitter_gain != 0:
#         split_count += 1
#     if voters.bundle_pass and voters.bundle_passage==[-1,-1]:
#         gerrymander_count += 1
#     if not voters.bundle_pass and voters.bundle_passage==[1,1]:
#         gerrymander_count += 1

#     splitter_gains.append(voters.splitter_gain)
#     nonsplitter_gains.append(voters.nonsplitter_gain)
#     bundle_maj.append(voters.bundle_maj_agreement)
#     split_maj.append(voters.split_bundle_maj_agreement)
#     splitter_counts.append(voters.splitter_count)
#     nonsplit_pass_counts.append(voters.nonsplit_pass)
#     nonsplit_fail_counts.append(voters.nonsplit_fail)
#     bundle_vi_pair_agree_list.append(voters.bundle_vi_pair_agreement)
#     split_bundle_vi_pair_agree_list.append(voters.split_bundle_vi_pair_agreement)
#     maj_vi_pair_agree_list.append(voters.maj_vi_pair_agreement)
    
#     vi_pair_scores.append((2*voters.split_bundle_vi_pair_agreement - voter_num*issue_num)/(voter_num*issue_num))
    
#     if voters.bundle_pass:
#         bundle_pass_count += 1
    

# print(f'Probability of a bundle split mattering is {split_count/iters}')
# print(f'Probability that both subbundles flip from bundle is {gerrymander_count/iters}')
# print(f'Expected gain for a splitter is {np.mean(splitter_gains)}')
# print(f'Expected gain for a non-splitter is {np.mean(nonsplitter_gains)}')
# print(f'Expected agreement between single bundle and majority rule is {np.mean(bundle_maj)}')
# print(f'Expected agreement between two bundles and majority rule is {np.mean(split_maj)}')

# print(f'Expected number of splitters is {np.mean(splitter_counts)}')
# print(f'Expected number of non-splitters that pass both subbundles is {np.mean(nonsplit_pass_counts)}')
# print(f'Expected number of non-splitters that fail both subbundles is {np.mean(nonsplit_fail_counts)}')
# print(f'Probability of a full bundle passing is {bundle_pass_count/iters}')

# print(f'Expected number of voter-issue pairs satisfied by majority rule is {np.mean(maj_vi_pair_agree_list)}')
# print(f'Expected number of voter-issue pairs satisfied by a single bundle is {np.mean(bundle_vi_pair_agree_list)}')
# print(f'Expected number of voter-issue pairs satisfied by a two bundles is {np.mean(split_bundle_vi_pair_agree_list)}')


# x_vals = [x/voter_num for x in splitter_counts] 
# y_vals = [(2*x - voter_num*issue_num)/(voter_num*issue_num) for x in bundle_vi_pair_agree_list]
# min_count = min(x_vals)
# max_count = max(x_vals)

# # reg = LinearRegression().fit(np.asarray(splitter_counts).reshape(-1,1), split_bundle_vi_pair_agree_list)
# reg = LinearRegression().fit(np.asarray(x_vals).reshape(-1,1), y_vals)
# intercept = reg.intercept_
# slope = reg.coef_[0]

# plt.subplots()
# plt.scatter(x_vals, y_vals, alpha=0.08)
# plt.plot([min_count, max_count],[intercept+slope*min_count, intercept+slope*max_count], c='r')
# plt.xlabel('Fraction of voters in favor of splitting')
# plt.ylabel('Single Bundle Utility Score')
# plt.tight_layout()
# plt.savefig('fig4d.png', dpi=300)



# x_vals = [x/voter_num for x in splitter_counts]
# y_vals = [(2*x - voter_num*issue_num)/(voter_num*issue_num) for x in split_bundle_vi_pair_agree_list]
# min_count = min(x_vals)
# max_count = max(x_vals)

# # reg = LinearRegression().fit(np.asarray(splitter_counts).reshape(-1,1), split_bundle_vi_pair_agree_list)
# reg = LinearRegression().fit(np.asarray(x_vals).reshape(-1,1), y_vals)
# intercept = reg.intercept_
# slope = reg.coef_[0]

# plt.subplots()
# plt.scatter(x_vals, y_vals, alpha=0.08)
# plt.plot([min_count, max_count],[intercept+slope*min_count, intercept+slope*max_count], c='r')
# plt.xlabel('Fraction of voters in favor of splitting')
# plt.ylabel('Subbundles Utility Score')
# plt.tight_layout()
# plt.savefig('fig4e.png', dpi=300)


# x_vals = [x/voter_num for x in splitter_counts]
# y_vals = [(2*split_bundle_vi_pair_agree_list[i]-voter_num*issue_num)/(voter_num*issue_num) - (2*bundle_vi_pair_agree_list[i]-voter_num*issue_num)/(voter_num*issue_num) for i in range(iters)]
# min_count = min(x_vals)
# max_count = max(x_vals)

# # reg = LinearRegression().fit(np.asarray(splitter_counts).reshape(-1,1), split_bundle_vi_pair_agree_list)
# reg = LinearRegression().fit(np.asarray(x_vals).reshape(-1,1), y_vals)
# intercept = reg.intercept_
# slope = reg.coef_[0]

# plt.subplots()
# plt.scatter(x_vals, y_vals, alpha=0.08)
# plt.plot([min_count, max_count],[intercept+slope*min_count, intercept+slope*max_count], c='r')
# plt.xlabel('Fraction of voters in favor of splitting')
# plt.ylabel('Utility Score Change from Split')
# plt.tight_layout()
# plt.savefig('fig4f.png', dpi=300)







###############################################################
# Measure gerrymanderability of single random voting profile
###############################################################

# voter_num = 15
# # dont go above 13 issues if measuring all partitions
# issue_num = 13
# voters = Voters(voter_num, issue_num,'iid', util_probs=[0,0.5,0.5,0])
# voters.gerry_measure()


# sub_sizes = voters.subset_sizes.copy()
# mms = [[] for _ in range(len(sub_sizes))]
# egs = [[] for _ in range(len(sub_sizes))]
# fracs = [[] for _ in range(len(sub_sizes))]
# subsets = [[] for _ in range(len(sub_sizes))]

# for i in range(len(voters.subsets)):
#     indx = sub_sizes.index(len(voters.subsets[i]))
#     mms[indx].append(voters.mean_median[i])
#     egs[indx].append(voters.eg[i])
#     fracs[indx].append(voters.nonmaj_frac[i])
#     subsets[indx].append(voters.subsets[i])
    

# plt.subplots()
# plt.hist(mms, stacked=True)
# plt.legend(sub_sizes)
# plt.xlabel('Mean-Median Score for a Subbundle of Given Size')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.tight_layout()

# plt.subplots()
# plt.hist(egs, stacked=True)
# plt.legend(sub_sizes)
# plt.xlabel('Efficiency Gap Score for a Subbundle of Given Size')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.tight_layout()


# # plt.subplots()
# # for i in range(len(fracs)):
# #     xvals = [j/sub_sizes[i] for j in range(sub_sizes[i]+1)]
# #     yvals = [0 for _ in range(sub_sizes[i]+1)]
# #     for frac in fracs[i]:
# #         indx = xvals.index(frac)
# #         yvals[indx] += 1
# #     for j in range(len(yvals)):
# #         yvals[j] = yvals[j]/len(fracs[i])
# #     plt.plot(xvals,yvals)
# # plt.legend(sub_sizes)
# # plt.xlabel('Fraction of subbundle passed/failed against the majority per subbundle size')
# # plt.ylabel('Probability')



# plt.subplots()
# for i in range(len(fracs)):
#     non_maj_fracs = [j/sub_sizes[i] for j in range(sub_sizes[i]+1)]
#     xvals = [1-2*x for x in non_maj_fracs]
#     yvals = [0 for _ in range(sub_sizes[i]+1)]
#     for frac in fracs[i]:
#         indx = non_maj_fracs.index(frac)
#         yvals[indx] += 1
#     plt.plot(xvals,yvals)
# plt.legend(sub_sizes)
# plt.xlabel('Subbundle Issue Score')
# plt.ylabel('Frequency')
# # plt.yscale('log')
# plt.tight_layout()



# partitions = [[[0]]]
# for i in range(1,issue_num):
#     new_partition = []
#     for part in partitions:
#         for j in range(len(part)):
#             new_partition.append(part[:j] + [part[j]+[i]] + part[j+1:])
#         new_partition.append(part+[[i]])
#     partitions = new_partition
# print('partitions done')

# odd_partitions = []
# for i in range(len(partitions)):
#     part = partitions[i]
#     odd_only = True
#     for sub in part:
#         if len(sub)%2==0:
#             odd_only = False
#             break
#     if odd_only:
#         odd_partitions.append(part)
# print('odd-only partitions done')

# # nonmaj_num_full = []
# # for part in odd_partitions:
# #     nonmaj_num = 0
# #     for sub in part:
# #         indx = voters.subsets.index(sub)
# #         nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
# #     nonmaj_num_full.append(nonmaj_num)
# issue_score_full = []
# for part in odd_partitions:
#     nonmaj_num = 0
#     for sub in part:
#         indx = voters.subsets.index(sub)
#         nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
#     issue_score_full.append((issue_num-2*nonmaj_num)/issue_num)


# plt.subplots()
# # plt.hist(nonmaj_num_full, bins=list(range(issue_num)))
# plt.hist(issue_score_full)
# plt.xlabel('Subbundle Scheme Issue Score')
# plt.ylabel('Frequency')
# # plt.yscale('log')
# plt.tight_layout()




###############################################
# gerrymandering with multiple utility profiles
###############################################

# iters =  100
# voter_num = 15
# # dont go above 13 issues if measuring all partitions
# issue_num = 13


# partitions = [[[0]]]
# for i in range(1,issue_num):
#     new_partition = []
#     for part in partitions:
#         for j in range(len(part)):
#             new_partition.append(part[:j] + [part[j]+[i]] + part[j+1:])
#         new_partition.append(part+[[i]])
#     partitions = new_partition
# print('partitions done')

# odd_partitions = []
# for i in range(len(partitions)):
#     part = partitions[i]
#     odd_only = True
#     for sub in part:
#         if len(sub)%2==0:
#             odd_only = False
#             break
#     if odd_only:
#         odd_partitions.append(part)
# print('odd-only partitions done')




# sub_sizes = list(range(1,issue_num+1, 2))
# mms = [[] for _ in range(len(sub_sizes))]
# egs = [[] for _ in range(len(sub_sizes))]
# fracs = [[] for _ in range(len(sub_sizes))]
# subsets = [[] for _ in range(len(sub_sizes))]
# issue_score_full = []
# issue_score_minimums = []

# for iter in range(iters):
#     print(iter)
    
#     issue_scores = []

#     voters = Voters(voter_num, issue_num,'iid', util_probs=[0,0.5,0.5,0])
#     voters.gerry_measure()

#     for i in range(len(voters.subsets)):
#         indx = sub_sizes.index(len(voters.subsets[i]))
#         mms[indx].append(voters.mean_median[i])
#         egs[indx].append(voters.eg[i])
#         fracs[indx].append(voters.nonmaj_frac[i])
#         subsets[indx].append(voters.subsets[i])
        
#     for part in odd_partitions:
#         nonmaj_num = 0
#         for sub in part:
#             indx = voters.subsets.index(sub)
#             nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
#         issue_score_full.append((issue_num-2*nonmaj_num)/issue_num)
#         issue_scores.append((issue_num-2*nonmaj_num)/issue_num)
    
#     issue_score_minimums.append(min(issue_scores))

# plt.subplots()
# plt.hist(mms, stacked=True)
# plt.legend(sub_sizes)
# plt.xlabel('Mean-Median Score of Subbundle')
# plt.ylabel('Frequency')
# # plt.yscale('log')
# plt.tight_layout()
# plt.savefig('fig5a.png', dpi=300)

# plt.subplots()
# plt.hist(mms, stacked=True)
# plt.legend(sub_sizes)
# plt.xlabel('Mean-Median Score of Subbundle')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig('fig5alog.png', dpi=300)

# plt.subplots()
# plt.hist(egs, stacked=True)
# plt.legend(sub_sizes)
# plt.xlabel('Efficiency Gap Score of Subbundle')
# plt.ylabel('Frequency')
# # plt.yscale('log')
# plt.tight_layout()
# plt.savefig('fig5b.png', dpi=300)

# plt.subplots()
# plt.hist(egs, stacked=True)
# plt.legend(sub_sizes)
# plt.xlabel('Efficiency Gap Score of Subbundle')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig('fig5blog.png', dpi=300)


# plt.subplots()
# for i in range(len(fracs)):
#     xvals = [j/sub_sizes[i] for j in range(sub_sizes[i]+1)]
#     yvals = [0 for _ in range(sub_sizes[i]+1)]
#     for frac in fracs[i]:
#         indx = xvals.index(frac)
#         yvals[indx] += 1
#     for j in range(len(yvals)):
#         yvals[j] = yvals[j]/len(fracs[i])
#     plt.plot(xvals,yvals)
# plt.legend(sub_sizes)
# plt.xlabel('Fraction of subbundle passed/failed against the majority per subbundle size')
# plt.ylabel('Probability')


# plt.subplots()
# for i in range(len(fracs)):
#     non_maj_fracs = [j/sub_sizes[i] for j in range(sub_sizes[i]+1)]
#     xvals = [1-2*x for x in non_maj_fracs]
#     yvals = [0 for _ in range(sub_sizes[i]+1)]
#     for frac in fracs[i]:
#         indx = non_maj_fracs.index(frac)
#         yvals[indx] += 1
#     plt.plot(xvals,yvals)
# plt.legend(sub_sizes)
# plt.xlabel('Subbundle Issue Score')
# plt.ylabel('Frequency')
# # plt.yscale('log')
# plt.tight_layout()
# plt.savefig('fig5c.png', dpi=300)

# plt.subplots()
# plt.hist(issue_score_full)
# plt.xlabel('Subbundle Scheme Issue Score')
# plt.ylabel('Frequency')
# # plt.yscale('log')
# plt.tight_layout()
# plt.savefig('fig5d.png', dpi=300)

# plt.subplots()
# plt.hist(issue_score_full)
# plt.xlabel('Subbundle Scheme Issue Score')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig('fig5dlog.png', dpi=300)


# # Set iters to 1000
# plt.subplots()
# plt.hist(issue_score_minimums, bins = [-6*0.9/6, -5*0.9/6, -4*0.9/6, -3*0.9/6, -2*0.9/6, -0.9/6,0])
# plt.xlabel('Minimum Subbundle Scheme Issue Score')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('fig5dnew.png', dpi=300)







##########################################################
##########################################################
# Supplementary Information
##########################################################
##########################################################


##########################################################
# Create plots demonstrating spatial model and parameter q
##########################################################
# voter_num = 21
# issue_num = 11
# q = 0.5

# voters = Voters(n=voter_num, m=issue_num, util_method='spatial', spatial_skew=q)
# voter_pos = voters.voter_positions
# voter_x = [v[0] for v in voter_pos]
# voter_y = [v[1] for v in voter_pos]
# issue_pos = voters.issue_positions
# issue_yes_x = [i[0][0] for i in issue_pos]
# issue_yes_y = [i[0][1] for i in issue_pos]
# issue_no_x = [i[1][0] for i in issue_pos]
# issue_no_y = [i[1][1] for i in issue_pos]

# plt.subplots()
# plt.scatter(voter_x, voter_y, c='k', label='Voters')
# plt.scatter(issue_yes_x, issue_yes_y, c='b', label='Issues: yes')
# plt.scatter(issue_no_x, issue_no_y, c='r', label='Issues: no')
# plt.legend()
# plt.title(f'$q={q}$')
# plt.tight_layout()
# plt.savefig('figSI1a.png', dpi=300)



# q = 0.85

# voters = Voters(n=voter_num, m=issue_num, util_method='spatial', spatial_skew=q)
# voter_pos = voters.voter_positions
# voter_x = [v[0] for v in voter_pos]
# voter_y = [v[1] for v in voter_pos]
# issue_pos = voters.issue_positions
# issue_yes_x = [i[1][0] for i in issue_pos]
# issue_yes_y = [i[1][1] for i in issue_pos]
# issue_no_x = [i[0][0] for i in issue_pos]
# issue_no_y = [i[0][1] for i in issue_pos]

# plt.subplots()
# plt.scatter(voter_x, voter_y, c='k', label='Voters')
# plt.scatter(issue_yes_x, issue_yes_y, c='b', label='Issues: yes')
# plt.scatter(issue_no_x, issue_no_y, c='r', label='Issues: no')
# plt.legend()
# plt.title(f'$q={q}$')
# plt.tight_layout()
# plt.savefig('figSI1b.png', dpi=300)


########################################################
# Changing the number of dimensions of the spatial model
########################################################
# voter_num = 101
# issue_num = 51
# iters = 1000

# bundle_vi_pair_agree_list = []
# maj_vi_pair_agree_list = []
# bundle_maj_issue_agree_list = []

# dims = list(range(1,11))

# for dim in dims:
#     print(dim)
#     bundle_vi_pair_agreements = []
#     maj_vi_pair_agreements = []
#     bundle_maj_issue_agreements = []
    
#     for _ in range(iters):
#         voters = Voters(n=voter_num, m=issue_num, util_method='spatial', spatial_dim=dim)
#         bundle_vi_pair_agreements.append(voters.bundle_vi_pair_agreement)
#         maj_vi_pair_agreements.append(voters.maj_vi_pair_agreement)
#         bundle_maj_issue_agreements.append(voters.bundle_maj_agreement)
        
#     bundle_vi_pair_agree_list.append(np.mean(bundle_vi_pair_agreements))
#     maj_vi_pair_agree_list.append(np.mean(maj_vi_pair_agreements))
#     bundle_maj_issue_agree_list.append(np.mean(bundle_maj_issue_agreements))

# plt.subplots()
# plt.plot(dims, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in bundle_vi_pair_agree_list])
# plt.plot(dims, [(2*x-voter_num*issue_num)/(voter_num*issue_num) for x in maj_vi_pair_agree_list])
# plt.xlabel('Number of Dimensions')
# plt.ylabel('Expected Utility Score')
# plt.legend(['Bundled Vote','Issue-by-Issue Vote'])
# plt.ylim((0,0.35))
# plt.tight_layout()
# plt.savefig('figSI2a.png', dpi=300)

# plt.subplots()
# plt.plot(dims, [(2*x-issue_num)/issue_num for x in bundle_maj_issue_agree_list])
# plt.plot([dims[0],dims[-1]], [1, 1])
# plt.xlabel('Number of Dimensions')
# plt.ylabel('Expected Issue Score')
# plt.legend(['Bundled Vote', 'Issue-by-Issue Vote'])
# plt.ylim((0,1.1))
# plt.tight_layout()
# plt.savefig('figSI2b.png', dpi=300)




#############################################
# Examining distribution of utilities for individual voters
# Symmetric distributions mean diminishing marginal utility will not change results
#############################################
# voter_num = 51
# issue_num = 51
# iters = 1000

# bundle_utils = []
# ibi_utils = []

# for _ in range(iters):
#     voters = Voters(voter_num, issue_num,'iid', util_probs=[0,0.5,0.5,0])
#     bundle_utils += voters.ind_util_bundle()
#     ibi_utils += voters.ind_util_issuebyissue()

# plt.subplots()
# plt.hist(bundle_utils)
# plt.subplots()
# plt.hist(ibi_utils)



