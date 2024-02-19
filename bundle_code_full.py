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
from voters_class_full import Voters
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
# plt.savefig('fig1a.png', dpi=300)

# plt.subplots()
# plt.plot(ps, [(2*x-issue_num)/issue_num for x in bundle_maj_issue_agree_list])
# plt.plot([0.5,ps[-1]], [1, 1])
# plt.xlabel('$p$')
# plt.ylabel('Expected Issue Score')
# plt.legend(['Bundled Vote', 'Issue-by-Issue Vote'])
# plt.tight_layout()
# plt.savefig('fig1b.png', dpi=300)


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
# plt.savefig('fig1c.png', dpi=300)

# plt.subplots()
# plt.plot(qs, [(2*x-issue_num)/issue_num for x in bundle_maj_issue_agree_list])
# plt.plot([qs[0],qs[-1]], [1, 1])
# plt.xlabel('$q$')
# plt.ylabel('Expected Issue Score')
# plt.legend(['Bundled Vote', 'Issue-by-Issue Vote'])
# plt.tight_layout()
# plt.savefig('fig1d.png', dpi=300)


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
# plt.savefig('fig2a.png', dpi=300)


# plt.subplots()
# plt.hist(maj_sup_scores)
# sim_mean_2 = np.mean(maj_sup_scores)
# plt.plot([sim_mean_2,sim_mean_2],[0,3500])
# plt.legend(['Simulation mean'])
# plt.xlabel('Issue Score')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('fig2b.png', dpi=300)



# vi_pair_scores = []
# maj_sup_scores = []

# for _ in range(iters):
#     voters = Voters(n=voter_num,m=issue_num,util_method='spatial')
#     vi_pair_scores.append((2*voters.bundle_vi_pair_agreement-(voter_num*issue_num))/(voter_num*issue_num))
#     maj_sup_scores.append((2*voters.bundle_maj_agreement-issue_num)/issue_num)

# plt.subplots()
# plt.hist(vi_pair_scores)
# sim_mean_1 = np.mean(vi_pair_scores)
# plt.plot([sim_mean_1,sim_mean_1],[0,3500])
# plt.legend(['Simulation mean'])
# plt.xlabel('Utility Score')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('fig2c.png', dpi=300)



# plt.subplots()
# plt.hist(maj_sup_scores)
# sim_mean_2 = np.mean(maj_sup_scores)
# plt.plot([sim_mean_2,sim_mean_2],[0,4000])
# plt.legend(['Simulation mean'])
# plt.xlabel('Issue Score')
# plt.ylabel('Frequency')
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
# plt.savefig('fig3.png', dpi=300)


########################################
# Create table of frequency of
# subbundle flipping entire vote
########################################

# voter_nums = [11, 101, 1001]
# issue_nums = [11, 101, 1001]

# for voter_num in voter_nums:
#     for issue_num in issue_nums:
#         iters = 100
#         samples = 100

#         iter_list = []
#         for iter in range(iters):
#             sample_list = []
#             for sample in range(samples):
#                 gerrymander_count = 0
#                 # if iter%1000==0:
#                 #     print(iter)
#                 voters = Voters(voter_num, issue_num, 'iid', util_probs=[0,0.5,0.5,0], bundle_split='random')
#                 if voters.bundle_pass and voters.bundle_passage==[-1,-1]:
#                     gerrymander_count += 1
#                 if not voters.bundle_pass and voters.bundle_passage==[1,1]:
#                     gerrymander_count += 1
#                 sample_list.append(gerrymander_count)
#             iter_list.append(st.tmean(sample_list))
        
#         prob = st.tmean(iter_list)
#         sd = st.tstd(iter_list)
                
#         print(f'n: {voter_num}, m: {issue_num}')
#         print(f'Probability that both subbundles flip from bundle is {prob} pm {sd}')


###############################################################
# Measure gerrymanderability of single random voting profile
###############################################################

# voter_num = 15
# # dont go above 13 issues if measuring all partitions
# issue_num = 11
# voters = Voters(voter_num, issue_num,'iid', util_probs=[0,0.5,0.5,0])
# voters.gerry_measure_all_subsets()


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

# issue_score_full = []
# for part in partitions:
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
# issue_num = 11


# partitions = [[[0]]]
# for i in range(1,issue_num):
#     new_partition = []
#     for part in partitions:
#         for j in range(len(part)):
#             new_partition.append(part[:j] + [part[j]+[i]] + part[j+1:])
#         new_partition.append(part+[[i]])
#     partitions = new_partition
# print('partitions done')


# sub_sizes = list(range(1,issue_num+1))
# mms = [[] for _ in range(len(sub_sizes))]
# egs = [[] for _ in range(len(sub_sizes))]
# fracs = [[] for _ in range(len(sub_sizes))]
# subsets = [[] for _ in range(len(sub_sizes))]
# issue_score_full = []
# issue_score_minimums = []

# for iter in range(iters):
#     if iter%10==0:
#         print(iter)
    
#     issue_scores = []

#     voters = Voters(voter_num, issue_num,'iid', util_probs=[0,0.5,0.5,0])
#     voters.gerry_measure_all_subsets()

#     for i in range(len(voters.subsets)):
#         indx = sub_sizes.index(len(voters.subsets[i]))
#         mms[indx].append(voters.mean_median[i])
#         egs[indx].append(voters.eg[i])
#         fracs[indx].append(voters.nonmaj_frac[i])
#         subsets[indx].append(voters.subsets[i])
        
#     for part in partitions:
#         nonmaj_num = 0
#         for sub in part:
#             indx = voters.subsets.index(sub)
#             nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
#         issue_score_full.append((issue_num-2*nonmaj_num)/issue_num)
#         issue_scores.append((issue_num-2*nonmaj_num)/issue_num)
    
#     issue_score_minimums.append(min(issue_scores))

# plt.subplots()
# plt.hist(egs, stacked=True)
# plt.legend(sub_sizes)
# plt.xlabel('Efficiency Gap Score of Subbundle')
# plt.ylabel('Frequency')
# # plt.yscale('log')
# plt.tight_layout()
# plt.savefig('fig4a.png', dpi=300)

# # plt.subplots()
# # plt.hist(egs, stacked=True)
# # plt.legend(sub_sizes)
# # plt.xlabel('Efficiency Gap Score of Subbundle')
# # plt.ylabel('Frequency')
# # plt.yscale('log')
# # plt.tight_layout()
# # plt.savefig('fig4alog.png', dpi=300)

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
# plt.savefig('fig4b.png', dpi=300)

# plt.subplots()
# plt.hist(issue_score_full, bins = [-10/11, -8/11, -6/11, -4/11, -2/11, 0, 2/11, 4/11, 6/11, 8/11, 10/11, 1])
# plt.xlabel('Subbundle Scheme Issue Score')
# plt.ylabel('Frequency')
# # plt.yscale('log')
# plt.tight_layout()
# plt.savefig('fig4c.png', dpi=300)

# # plt.subplots()
# # plt.hist(issue_score_full)
# # plt.xlabel('Subbundle Scheme Issue Score')
# # plt.ylabel('Frequency')
# # plt.yscale('log')
# # plt.tight_layout()
# # plt.savefig('fig4clog.png', dpi=300)


# iters = 1000

# issue_score_minimums = []

# for iter in range(iters):
#     if iter%100==0:
#         print(iter)
    
#     issue_scores = []

#     voters = Voters(voter_num, issue_num,'iid', util_probs=[0,0.5,0.5,0])
#     voters.gerry_measure_all_subsets()

        
#     for part in partitions:
#         nonmaj_num = 0
#         for sub in part:
#             indx = voters.subsets.index(sub)
#             nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
#         issue_scores.append((issue_num-2*nonmaj_num)/issue_num)
    
#     issue_score_minimums.append(min(issue_scores))


# # Set iters to 1000
# plt.subplots()
# plt.hist(issue_score_minimums, bins = [-10/11, -8/11, -6/11, -4/11, -2/11, 0, 2/11, 4/11])
# plt.xlabel('Minimum Subbundle Scheme Issue Score')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('fig4d.png', dpi=300)







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
# plt.savefig('figS1a.png', dpi=300)



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
# plt.savefig('figS1b.png', dpi=300)


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
# plt.savefig('figS2a.png', dpi=300)

# plt.subplots()
# plt.plot(dims, [(2*x-issue_num)/issue_num for x in bundle_maj_issue_agree_list])
# plt.plot([dims[0],dims[-1]], [1, 1])
# plt.xlabel('Number of Dimensions')
# plt.ylabel('Expected Issue Score')
# plt.legend(['Bundled Vote', 'Issue-by-Issue Vote'])
# plt.ylim((0,1.1))
# plt.tight_layout()
# plt.savefig('figS2b.png', dpi=300)


######################################################
# Recreating IID figs with uniform dist on [-2,2]
######################################################
#########################################
# Create plots when support/oppose becomes unbalanced
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
#         voters = Voters(n=voter_num, m=issue_num, util_method='uniform', uni_skew = p)
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
# plt.savefig('figS4a.png', dpi=300)

# plt.subplots()
# plt.plot(ps, [(2*x-issue_num)/issue_num for x in bundle_maj_issue_agree_list])
# plt.plot([0.5,ps[-1]], [1, 1])
# plt.xlabel('$p$')
# plt.ylabel('Expected Issue Score')
# plt.legend(['Bundled Vote', 'Issue-by-Issue Vote'])
# plt.tight_layout()
# plt.savefig('figS4b.png', dpi=300)


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
#     voters = Voters(n=voter_num,m=issue_num,util_method='uniform')
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
# plt.savefig('figS5a.png', dpi=300)


# plt.subplots()
# plt.hist(maj_sup_scores)
# sim_mean_2 = np.mean(maj_sup_scores)
# plt.plot([sim_mean_2,sim_mean_2],[0,3500])
# plt.legend(['Simulation mean'])
# plt.xlabel('Issue Score')
# plt.ylabel('Frequency')
# plt.tight_layout()
# plt.savefig('figS5b.png', dpi=300)

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
#         voters = Voters(n=voter_num, m=issue_num, util_method='uniform', bundle_split='random', bundle_num=num)
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
# plt.savefig('figS6.png', dpi=300)


###############################################
# gerrymandering with multiple utility profiles
###############################################

iters =  100
voter_num = 15
# dont go above 13 issues if measuring all partitions
issue_num = 11


partitions = [[[0]]]
for i in range(1,issue_num):
    new_partition = []
    for part in partitions:
        for j in range(len(part)):
            new_partition.append(part[:j] + [part[j]+[i]] + part[j+1:])
        new_partition.append(part+[[i]])
    partitions = new_partition
print('partitions done')


sub_sizes = list(range(1,issue_num+1))
mms = [[] for _ in range(len(sub_sizes))]
egs = [[] for _ in range(len(sub_sizes))]
fracs = [[] for _ in range(len(sub_sizes))]
subsets = [[] for _ in range(len(sub_sizes))]
issue_score_full = []
issue_score_minimums = []

for iter in range(iters):
    if iter%10==0:
        print(iter)
    
    issue_scores = []

    voters = Voters(voter_num, issue_num,'uniform')
    voters.gerry_measure_all_subsets()

    for i in range(len(voters.subsets)):
        indx = sub_sizes.index(len(voters.subsets[i]))
        mms[indx].append(voters.mean_median[i])
        egs[indx].append(voters.eg[i])
        fracs[indx].append(voters.nonmaj_frac[i])
        subsets[indx].append(voters.subsets[i])
        
    for part in partitions:
        nonmaj_num = 0
        for sub in part:
            indx = voters.subsets.index(sub)
            nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
        issue_score_full.append((issue_num-2*nonmaj_num)/issue_num)
        issue_scores.append((issue_num-2*nonmaj_num)/issue_num)
    
    issue_score_minimums.append(min(issue_scores))

plt.subplots()
plt.hist(egs, stacked=True)
plt.legend(sub_sizes)
plt.xlabel('Efficiency Gap Score of Subbundle')
plt.ylabel('Frequency')
# plt.yscale('log')
plt.tight_layout()
plt.savefig('figS7a.png', dpi=300)

# plt.subplots()
# plt.hist(egs, stacked=True)
# plt.legend(sub_sizes)
# plt.xlabel('Efficiency Gap Score of Subbundle')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig('figS7alog.png', dpi=300)

plt.subplots()
for i in range(len(fracs)):
    non_maj_fracs = [j/sub_sizes[i] for j in range(sub_sizes[i]+1)]
    xvals = [1-2*x for x in non_maj_fracs]
    yvals = [0 for _ in range(sub_sizes[i]+1)]
    for frac in fracs[i]:
        indx = non_maj_fracs.index(frac)
        yvals[indx] += 1
    plt.plot(xvals,yvals)
plt.legend(sub_sizes)
plt.xlabel('Subbundle Issue Score')
plt.ylabel('Frequency')
# plt.yscale('log')
plt.tight_layout()
plt.savefig('figS7b.png', dpi=300)

plt.subplots()
plt.hist(issue_score_full, bins = [-10/11, -8/11, -6/11, -4/11, -2/11, 0, 2/11, 4/11, 6/11, 8/11, 10/11, 1])
plt.xlabel('Subbundle Scheme Issue Score')
plt.ylabel('Frequency')
# plt.yscale('log')
plt.tight_layout()
plt.savefig('figS7c.png', dpi=300)

# plt.subplots()
# plt.hist(issue_score_full)
# plt.xlabel('Subbundle Scheme Issue Score')
# plt.ylabel('Frequency')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig('figS7clog.png', dpi=300)


iters = 1000

issue_score_minimums = []

for iter in range(iters):
    if iter%100==0:
        print(iter)
    
    issue_scores = []

    voters = Voters(voter_num, issue_num,'uniform')
    voters.gerry_measure_all_subsets()

        
    for part in partitions:
        nonmaj_num = 0
        for sub in part:
            indx = voters.subsets.index(sub)
            nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
        issue_scores.append((issue_num-2*nonmaj_num)/issue_num)
    
    issue_score_minimums.append(min(issue_scores))


# Set iters to 1000
plt.subplots()
plt.hist(issue_score_minimums, bins = [-1,-10/11, -8/11, -6/11, -4/11, -2/11, 0, 2/11, 4/11])
plt.xlabel('Minimum Subbundle Scheme Issue Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('figS7d.png', dpi=300)
