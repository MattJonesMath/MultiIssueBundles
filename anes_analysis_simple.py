####################################
# Analyze data from ANES 2020 survey
####################################

import csv
from voters_class import Voters
import matplotlib.pyplot as plt
import random as rand
import multiprocessing
import json
import copy




# t1 is v202225 -> limits on campaign spending
#0 t2 is v202231x -> limits on imports
#1 t3 is v202232 -> change in immigration levels
#2 t4 is v202252x -> preferential hiring for blacks
#3 t5 is v202256 -> level of government regulation
#4 t6 is v202259x -> government work to reduce income inequality
#5 t7 is v202325 -> tax on millionaires
# t8 is v202331x -> vaccine requirements in schools
#6 t9 is v202336x -> regulation of greenhouse gases
# t10 is v202341x -> background checks for guns
#7 t11 is v202344x -> ban assault style weapons
# t12 is v202350x -> government action on opioid addiction
#8 t13 is v202361x -> free trade agreements
#9 t14 is v202376x -> universal basic income (12k/year)
#10 t15 is v202380x -> government spending on healthcare
voter_utilities = []
reps = 0
dems = 0
others = 0
dem_utils = []
rep_utils = []
other_utils = []


with open('anes_timeseries_2020_csv_20220210.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        utils = []
        
        repdem = int(row['V202429'])
        if repdem == 1:
            dems += 1
            voter_utilities = dem_utils
        elif repdem == 5:
            reps += 1
            voter_utilities = rep_utils
        else:
            others += 1
            voter_utilities = other_utils
            
        
        
        # t1 = int(row['V202225'])
        # if t1 == 1:
        #     utils.append(1)
        # elif t1 == 2:
        #     utils.append(-1)
        # elif t1 == 3:
        #     utils.append(0)
        # else:
        #     utils.append('x')
            
        t2 = int(row['V202231x'])
        if t2 in [1,2]:
            utils.append(1)
        elif t2 in [3,4]:
            utils.append(-1)
        else:
            utils.append('x')
            
        t3 = int(row['V202232'])
        if t3 in [1,2]:
            utils.append(1)
        elif t3 in [4,5]:
            utils.append(-1)
        elif t3 == 3:
            utils.append(0)
        else:
            utils.append('x')
            
        t4 = int(row['V202252x'])
        if t4 in [1,2]:
            utils.append(1)
        elif t4 in [3,4]:
            utils.append(-1)
        else:
            utils.append('x')
            
        t5 = int(row['V202256'])
        if t5 in [1,2,3]:
            utils.append(1)
        elif t5 in [5,6,7]:
            utils.append(-1)
        elif t5 == 4:
            utils.append(0)
        else:
            utils.append('x')
            
        t6 = int(row['V202259x'])
        if t6 in [1,2,3]:
            utils.append(1)
        elif t6 in [5,6,7]:
            utils.append(-1)
        elif t6 == 4:
            utils.append(0)
        else:
            utils.append('x')
            
        t7 = int(row['V202325'])
        if t7 == 1:
            utils.append(1)
        elif t7 == 2:
            utils.append(-1)
        elif t7 == 3:
            utils.append(0)
        else:
            utils.append('x')

        # comment out this one to reach 11            
        # t8 = int(row['V202331x'])
        # if t8 in [1,2,3]:
        #     utils.append(1)
        # elif t8 in [5,6,7]:
        #     utils.append(-1)
        # elif t8 == 4:
        #     utils.append(0)
        # else:
        #     utils.append('x')
            
        t9 = int(row['V202336x'])
        if t9 in [1,2,3]:
            utils.append(1)
        elif t9 in [5,6,7]:
            utils.append(-1)
        elif t9 == 4:
            utils.append(0)
        else:
            utils.append('x')
            
        # t10 = int(row['V202341x'])
        # if t10 in [1,2,3]:
        #     utils.append(1)
        # elif t10 in [5,6,7]:
        #     utils.append(-1)
        # elif t10 == 4:
        #     utils.append(0)
        # else:
        #     utils.append('x')
            
        t11 = int(row['V202344x'])
        if t11 in [1,2,3]:
            utils.append(1)
        elif t11 in [5,6,7]:
            utils.append(-1)
        elif t11 == 4:
            utils.append(0)
        else:
            utils.append('x')

        # comment out this one to reach 11            
        # t12 = int(row['V202350x'])
        # if t12 in [1,2,3]:
        #     utils.append(1)
        # elif t12 in [5,6,7]:
        #     utils.append(-1)
        # elif t12 == 4:
        #     utils.append(0)
        # else:
        #     utils.append('x')
            
        t13 = int(row['V202361x'])
        if t13 in [1,2,3]:
            utils.append(1)
        elif t13 in [5,6,7]:
            utils.append(-1)
        elif t13 == 4:
            utils.append(0)
        else:
            utils.append('x')
            
        t14 = int(row['V202376x'])
        if t14 in [1,2,3]:
            utils.append(1)
        elif t14 in [5,6,7]:
            utils.append(-1)
        elif t14 == 4:
            utils.append(0)
        else:
            utils.append('x')
            
        t15 = int(row['V202380x'])
        if t15 in [1,2,3]:
            utils.append(1)
        elif t15 in [5,6,7]:
            utils.append(-1)
        elif t15 == 4:
            utils.append(0)
        else:
            utils.append('x')

        voter_utilities.append(utils)


# voter_utilities_clean= []
# for util in voter_utilities:
#     if 'x' not in util and 0 not in util:
#         voter_utilities_clean.append(util)
        
# # n = len(utils_full)
# n = len(voter_utilities_clean)
# m = len(voter_utilities[0])
# voters = Voters(n, m, util_method='import', util_import=voter_utilities_clean)

# # issues = []
# # for i in range(m):
# #     issue_utils = [utils_full[j][i] for j in range(n)]
# #     issues.append(issue_utils)
    
# issues_full = []
# for i in range(m):
#     issue_utils = [voter_utilities_clean[j][i] for j in range(n)]
#     issues_full.append(issue_utils)
    
# print([x.count(1)-n/2 for x in issues_full])
    
# voters = Voters(n,m, util_method='import', util_import=voter_utilities_clean)


# print('Data done')
 
 
# partitions = [[[0]]]
# for i in range(1,m):
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



# sub_sizes = list(range(1,m+1, 2))

# issue_scores = []

# voters.gerry_measure()
    
# for part in odd_partitions:
#     nonmaj_num = 0
#     for sub in part:
#         indx = voters.subsets.index(sub)
#         nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
#     issue_scores.append((m-2*nonmaj_num)/m)

# print(min(issue_scores))

# plt.subplots()
# plt.hist(issue_scores, bins = [-1, -12/13, -10/13, -8/13, -6/13, -4/13, -2/13, 0, 2/13, 4/13, 6/13, 8/13, 10/13, 12/13, 1])
# plt.xlabel('Subbundle Scheme Issue Score')
# plt.ylabel('Frequency')
# plt.tight_layout()


##################################
# Sample population of voters
##################################
dem_utils_clean= []
for util in dem_utils:
    if 'x' not in util and 0 not in util:
    # if 'x' not in util:
        dem_utils_clean.append(util)

rep_utils_clean= []
for util in rep_utils:
    if 'x' not in util and 0 not in util:
    # if 'x' not in util:
        rep_utils_clean.append(util)

other_utils_clean= []
for util in other_utils:
    if 'x' not in util and 0 not in util:
    # if 'x' not in util:
        other_utils_clean.append(util)

print('Data done')

m = len(other_utils[0])

partitions = [[[0]]]
for i in range(1,m):
    new_partition = []
    for part in partitions:
        for j in range(len(part)):
            new_partition.append(part[:j] + [part[j]+[i]] + part[j+1:])
        new_partition.append(part+[[i]])
    partitions = new_partition
print('partitions done')

odd_partitions = []
for i in range(len(partitions)):
    part = partitions[i]
    odd_only = True
    for sub in part:
        if len(sub)%2==0:
            odd_only = False
            break
    if odd_only:
        odd_partitions.append(part)
print('odd-only partitions done')

flips = [[1],[-1]]
while len(flips[0])<m:
    new_flips = []
    for flip in flips:
        new_flips.append(flip+[1])
        new_flips.append(flip+[-1])
    flips = new_flips


n = 31
rand.shuffle(dem_utils_clean)
rand.shuffle(rep_utils_clean)
rand.shuffle(other_utils_clean)
# utilities = dem_utils_clean[:10]+rep_utils_clean[:10]+other_utils_clean[:11]
utilities = [[-1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [-1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1], [-1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1], [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [-1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1], [1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1], [1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1], [1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1], [1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1], [1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1], [1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1], [-1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1], [-1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1], [-1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1], [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [-1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1], [-1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1], [1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1], [-1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1], [-1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1], [1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1], [1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1], [-1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1]]

min_issue_scores = []
min_part_indx = []
# for flip in flips:
for i in range(len(flips)):
    flip = flips[i]
    if i%200==0:
        print(i)
        
    utilities_copy = copy.deepcopy(utilities)
    for i in range(m):
        for j in range(n):
            utilities_copy[j][i] *= flip[i]
    
    voters = Voters(n, m, util_method='import', util_import=utilities_copy)
    issue_scores = []
    voters.gerry_measure()        
    for part in odd_partitions:
        nonmaj_num = 0
        for sub in part:
            indx = voters.subsets.index(sub)
            nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
        issue_scores.append((m-2*nonmaj_num)/m)
        
        if (m-2*nonmaj_num)/m==-5/11:
            print(flip)
            print(part)
            for sub in part:
                indx = voters.subsets.index(sub)
                print(f'Subbundle {sub} has support {voters.subset_votes[indx]}')
                for issue in sub:
                    print(f'Issue {issue} has support {voters.voter_support[issue]}')
                # print(voters.nonmaj_frac[indx]*len(sub))
            break
        
    
    min_issue_scores.append(min(issue_scores))
    min_part_indx.append(issue_scores.index(min(issue_scores)))

print(min(min_issue_scores))
# voters = Voters(n,m,util_method='import', util_import=utilities)
# issue_scores = []
# voters.gerry_measure()        
# for part in odd_partitions:
#     nonmaj_num = 0
#     for sub in part:
#         indx = voters.subsets.index(sub)
#         nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
#     issue_scores.append((m-2*nonmaj_num)/m)

# print(min(issue_scores))