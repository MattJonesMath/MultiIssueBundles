####################################
# Analyze data from ANES 2020 survey
####################################

import csv
from voters_class_full import Voters
import matplotlib.pyplot as plt
import random as rand
import multiprocessing
import json





# t1 is v202225 -> limits on campaign spending
# t2 is v202231x -> limits on imports
# t3 is v202232 -> change in immigration levels
# t4 is v202252x -> preferential hiring for blacks
# t5 is v202256 -> level of government regulation
# t6 is v202259x -> government work to reduce income inequality
# t7 is v202325 -> tax on millionaires
# t8 is v202331x -> vaccine requirements in schools
# t9 is v202336x -> regulation of greenhouse gases
# t10 is v202341x -> background checks for guns
# t11 is v202344x -> ban assault style weapons
# t12 is v202350x -> government action on opioid addiction
# t13 is v202361x -> free trade agreements
# t14 is v202376x -> universal basic income (12k/year)
# t15 is v202380x -> government spending on healthcare



##################################
# random flip of issues
# multiprocessing for speed
##################################



def sim(param):
    flip = param[0]
    utilities_copy = param[1]
    partitions = param[2]
    
    n = len(utilities_copy)
    m = len(utilities_copy[0])
    # utilities_copy = voter_utilities_clean.copy()
    for i in range(m):
        for j in range(n):
            utilities_copy[j][i] *= flip[i]

    voters = Voters(n, m, util_method='import', util_import=utilities_copy)
    issue_scores = []
    voters.gerry_measure_all_subsets()        
    for part in partitions:
        nonmaj_num = 0
        for sub in part:
            indx = voters.subsets.index(sub)
            nonmaj_num += voters.nonmaj_frac[indx]*len(sub)
        issue_scores.append((m-2*nonmaj_num)/m)
    
    indx = issue_scores.index(min(issue_scores))
    
    return [issue_scores[indx], partitions[indx], flip]


if __name__ == '__main__':

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
                
            # t2 = int(row['V202231x'])
            # if t2==1:
            #     utils.append(1)
            # elif t2==2:
            #     utils.append(0.5)
            # elif t2==3:
            #     utils.append(-0.5)
            # elif t2==4:
            #     utils.append(-1)
            # else:
            #     utils.append('x')
                
            # t3 = int(row['V202232'])
            # if t3 in [1,2,3,4,5]:
            #     utils.append(1.5-0.5*t3)
            # else:
            #     utils.append('x')
                
            # t4 = int(row['V202252x'])
            # if t4==1:
            #     utils.append(1)
            # elif t4==2:
            #     utils.append(0.5)
            # elif t4==3:
            #     utils.append(-0.5)
            # elif t4==4:
            #     utils.append(-1)
            # else:
            #     utils.append('x')
                
            t5 = int(row['V202256'])
            if t5 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t5/3)
            else:
                utils.append('x')
                
            t6 = int(row['V202259x'])
            if t6 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t6/3)
            else:
                utils.append('x')
                
            # t7 = int(row['V202325'])
            # if t7 in [1,2,3]:
            #     utils.append(2-t7)
            # else:
            #     utils.append('x')
            
            t8 = int(row['V202331x'])
            if t8 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t8/3)
            else:
                utils.append('x')
                
            t9 = int(row['V202336x'])
            if t9 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t9/3)
            else:
                utils.append('x')
                
            t10 = int(row['V202341x'])
            if t10 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t10/3)
            else:
                utils.append('x')
                
            t11 = int(row['V202344x'])
            if t11 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t11/3)
            else:
                utils.append('x')
    
            t12 = int(row['V202350x'])
            if t12 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t12/3)
            else:
                utils.append('x')
                
            t13 = int(row['V202361x'])
            if t13 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t13/3)
            else:
                utils.append('x')
                
            t14 = int(row['V202376x'])
            if t14 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t14/3)
            else:
                utils.append('x')
                
            t15 = int(row['V202380x'])
            if t15 in [1,2,3,4,5,6,7]:
                utils.append(4/3-t15/3)
            else:
                utils.append('x')
    
            voter_utilities.append(utils)
    
    # voter_utilities_clean= []
    # for util in voter_utilities:
    #     if 'x' not in util and 0 not in util:
    #         voter_utilities_clean.append(util)
    
    # n = len(voter_utilities_clean)
    # m = len(voter_utilities_clean[0])
    
    
    # issues_full = []
    # for i in range(m):
    #     issue_utils = [voter_utilities_clean[j][i] for j in range(n)]
    #     issues_full.append(issue_utils)
        
    # print([x.count(1)-n/2 for x in issues_full])
    
    ##################################
    # Sample population of voters
    ##################################
    dem_utils_clean= []
    for util in dem_utils:
        # if 'x' not in util and 0 not in util:
        if 'x' not in util:
            dem_utils_clean.append(util)

    rep_utils_clean= []
    for util in rep_utils:
        # if 'x' not in util and 0 not in util:
        if 'x' not in util:
            rep_utils_clean.append(util)

    other_utils_clean= []
    for util in other_utils:
        # if 'x' not in util and 0 not in util:
        if 'x' not in util:
            other_utils_clean.append(util)

    print('Data done')

    m = len(other_utils[0])
    n = 31
    
    partitions = [[[0]]]
    for i in range(1,m):
        new_partition = []
        for part in partitions:
            for j in range(len(part)):
                new_partition.append(part[:j] + [part[j]+[i]] + part[j+1:])
            new_partition.append(part+[[i]])
        partitions = new_partition
    print('partitions done')
    
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
    
    flips = [[1],[-1]]
    while len(flips[0])<m:
        new_flips = []
        for flip in flips:
            new_flips.append(flip+[1])
            new_flips.append(flip+[-1])
        flips = new_flips
    
    iters = 1000
    big_results = []
    utils_list = []
    pool = multiprocessing.Pool(processes=20)
    # pool = multiprocessing.Pool(processes=5)
    
    print('Beginning samples')
    for i in range(iters):
        if i%50==0:
            print(i)
        rand.shuffle(dem_utils_clean)
        rand.shuffle(rep_utils_clean)
        rand.shuffle(other_utils_clean)
        utilities = dem_utils_clean[:10]+rep_utils_clean[:10]+other_utils_clean[:11]
        utils_list.append(utilities)
        
        params = [[flip, utilities, partitions] for flip in flips]
        result = pool.map(sim, params)
        big_results.append(result)
    
    print('Samples completed')
    min_score = 1
    min_partition = []
    min_flip = []
    for l1 in big_results:
        for l2 in l1:
            if l2[0]<min_score:
                min_indx = big_results.index(l1)
                # print(big_results.index(l1),l1.index(l2))
                min_score = l2[0]
                min_partition = l2[1].copy()
                min_flip = l2[2].copy()
                
    print(min_score)
    print(min_partition)
    print(min_flip)
    
    mins_by_sample = [min([l2[0] for l2 in l1]) for l1 in big_results]
    
    # with open('data.json', 'w') as f:
    #     json.dump(result, f)
    

    #########
    plt.subplots()
    # plt.hist(mins_by_sample,bins = [-1, -8/9, -6/9, -4/9, -2/9, 0, 2/9, 4/9, 6/9, 8/9, 10/11, 1])
    plt.hist(mins_by_sample,bins = [-1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
    
    
    
    ###########
    # Save figure
    
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
    
    
    plt.subplots()
    plt.hist(mins_by_sample,bins = [-1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3])
    plt.xlabel('Issue Score of Subbundle Scheme')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('FigSI8.png', dpi=300)
    
    utils = utils_list[min_indx]
    for part in min_partition:
        for issue in part:
            yes_votes = 0
            for i in range(31):
                if utils[i][issue]*min_flip[issue]>0:
                    yes_votes += 1
                elif utils[i][issue]==0:
                    yes_votes += 0.5
            print(f'Issue {issue} has {yes_votes} votes')
        subbundle_votes = 0
        for i in range(31):
            util = 0
            for issue in part:
                util += utils[i][issue]*min_flip[issue]
            if util>0:
                subbundle_votes += 1
            elif util == 0:
                subbundle_votes += 0.5
        print(f'Subbundle {part} has {subbundle_votes} votes')