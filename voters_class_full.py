#########################
# Class: voters
# Creates utilities, defines functions to analyze
#########################

import random as rand
import numpy as np
import math
from itertools import combinations

class Voters:
    def __init__(self, n, m, util_method, util_import = [], utils = [-2,-1,1,2], util_probs = [0.25,0.25,0.25,0.25], spatial_dim=2, spatial_skew=0.5, trade_net='complete', trade_thresh=1, bundle_split='random', bundle_num=2, uni_skew = 0.5):
        self.n = n
        self.m = m
        self.util_method = util_method
        self.util_import = util_import
        self.utils = utils
        self.util_probs = util_probs
        self.spatial_dim = spatial_dim
        self.trade_net = trade_net
        self.trade_thresh = trade_thresh
        self.bundle_split = bundle_split
        self.spatial_skew = spatial_skew
        self.bundle_num=bundle_num
        self.uni_skew = uni_skew
        
        self.create_utilities()
        self.maj_util_vals()
        
        self.maj_rule_vote()
        # self.util_rule_vote()
        self.bundle_vote()
        # self.vote_post_trades()
        self.split_bundle()
        
        # self.print_all_info()
        
        
        
#####################
# Create utilities
#####################
    def create_utilities(self):
        if self.util_method == 'iid':
            self.utilities = [rand.choices(self.utils, weights=self.util_probs, k=self.m) for _ in range(self.n)]
            
        elif self.util_method == 'spatial':
            self.utilities = [[] for _ in range(self.n)]
            # self.voter_positions = [[rand.random() for _ in range(self.spatial_dim)] for _ in range(self.n)]
            # self.issue_positions = [[[rand.random() for _ in range(self.spatial_dim)], [rand.random() for _ in range(self.spatial_dim)]] for _ in range(self.m)]
            
            # Create skewed population and issues along all or just the first dimension
            self.voter_positions = []
            for _ in range(self.n):
                # skews all dimensions
                temp = []
                for _ in range(self.spatial_dim):
                    if rand.random()<self.spatial_skew:
                        temp.append(0.5*rand.random())
                    else:
                        temp.append(0.5+0.5*rand.random())
                self.voter_positions.append(temp)
                
                # only skews first dimension
                # temp = [rand.random() for _ in range(self.spatial_dim)]
                # if rand.random()<0.5+self.spatial_skew:
                #     temp[0] = temp[0]*0.5
                # else:
                #     temp[0] = 0.5 + temp[0]*0.5
                # self.voter_positions.append(temp)
                
            self.issue_positions = []
            for _ in range(self.m):
                # skews all dimensions
                temp_yes = []
                for _ in range(self.spatial_dim):
                    if rand.random()<self.spatial_skew:
                        temp_yes.append(0.5*rand.random())
                    else:
                        temp_yes.append(0.5+0.5*rand.random())
                temp_no = []
                for _ in range(self.spatial_dim):
                    if rand.random()<self.spatial_skew:
                        temp_no.append(0.5+0.5*rand.random())
                    else:
                        temp_no.append(0.5*rand.random())
                self.issue_positions.append([temp_no,temp_yes])
                
                
                # skews first dimension
                # pos1 = [rand.random() for _ in range(self.spatial_dim)]
                # pos2 = [rand.random() for _ in range(self.spatial_dim)]
                # if bool(rand.random()<0.5+self.spatial_skew) != bool(pos1[0]<pos2[0]):
                #     self.issue_positions.append([pos1,pos2])
                # else:
                #     self.issue_positions.append([pos2,pos1])
                        
            
            for issue in self.issue_positions:
                no_pos = np.asarray(issue[0])
                yes_pos = np.asarray(issue[1])
                # Utility is difference in distances from yes and no positions
                for i in range(self.n):
                    v_pos = np.asarray(self.voter_positions[i])
                    self.utilities[i].append(self.dist(v_pos,no_pos)-self.dist(v_pos,yes_pos))
         
        elif self.util_method == 'uniform':
            self.utilities = [[] for _ in range(self.n)]
            for i in range(self.n):
                for _ in range(self.m):
                    if rand.random()<self.uni_skew:
                        self.utilities[i].append(2*rand.random())
                    else:
                        self.utilities[i].append(-2*rand.random())
            # self.utilities = [[-2+4*rand.random() for _ in range(self.m)] for _ in range(self.n)]
            
        elif self.util_method == 'import':
            self.utilities = self.util_import
            
        else:
            raise ValueError('Unacceptable utility distribution method')
            

#####################
# Analyses
#####################

# Computes the number of votes and the utility for each issue
# Automatically runs when class is created
    def maj_util_vals(self):
        self.voter_support = []
        self.util_value = []
        for i in range(self.m):
            ind_votes = 0
            total_util = 0
            for j in range(self.n):
                total_util += self.utilities[j][i]
                if self.utilities[j][i]>0:
                    ind_votes += 1
                ########################
                # added for real data
                if self.utilities[j][i]==0:
                    ind_votes += 0.5
                ########################
            self.voter_support.append(ind_votes)
            self.util_value.append(total_util)

# Computes the net value of majority rule and the number of issue decisions passed with majority consent
    def maj_rule_vote(self):
        self.maj_rule_value = 0
        self.maj_rule_sup_issues = 0
        self.maj_vi_pair_agreement = 0
        
        self.maj_rule_passage = []
        
        for i in range(self.m):
            if self.voter_support[i]>self.n/2:
                self.maj_rule_passage.append(True)
                self.maj_rule_sup_issues += 1
                self.maj_rule_value += self.util_value[i]
            else:
                self.maj_rule_passage.append(False)
                self.maj_rule_value -= self.util_value[i]
                
        for i in range(self.m):
            if self.voter_support[i]>self.n/2:
                self.maj_vi_pair_agreement += self.voter_support[i]
            else:
                self.maj_vi_pair_agreement += self.n - self.voter_support[i]
                
# Computes the net value of utility rule and the number of issue decisions passed with majority consent
    def util_rule_vote(self):
        self.util_rule_value = 0
        self.util_rule_sup_issues = 0
        self.util_maj_agreement = 0
        
        for i in range(self.m):
            if self.util_value[i]>0:
                self.util_rule_value += self.util_value[i]
                self.util_rule_sup_issues += 1
                if self.voter_support[i]>self.n/2:
                    self.util_maj_agreement += 1
            elif self.util_value[i]<0:
                self.util_rule_value -= self.util_value[i]
                if self.voter_support[i]<self.n/2:
                    self.util_maj_agreement += 1
            else:
                self.util_rule_sup_issues += 1/2
                self.util_maj_agreement += 1/2

# Computes the net value of a bundled vote and the number of issue decisions passed with majority consent
    def bundle_vote(self):
        self.bundle_value = 0
        self.bundle_sup_issues = 0
        self.bundle_maj_agreement = 0
        self.bundle_vi_pair_agreement = 0
        
        bundle_votes = 0
        for i in range(self.n):
            total_util = 0
            for j in range(self.m):
                total_util += self.utilities[i][j]
            self.bundle_value += total_util
            if total_util>0:
                bundle_votes += 1
        
        if bundle_votes > self.n/2:
            self.bundle_pass = True
            self.bundle_sup_issues = self.m
            for vote_num in self.voter_support:
                self.bundle_vi_pair_agreement += vote_num
                if vote_num > self.n/2:
                    self.bundle_maj_agreement+=1

                    
        else:
            self.bundle_pass = False
            self.bundle_value = self.bundle_value*-1
            for vote_num in self.voter_support:
                self.bundle_vi_pair_agreement += (self.n-vote_num)
                if vote_num < self.n/2:
                    self.bundle_maj_agreement+=1
                    
        # self.bundle_vi_pair_agreement = (self.bundle_value+self.n*self.m)/2
            
# Create vote trading network
    def create_network(self):
        self.trade_network = []
        
        if self.trade_net=='complete':
            for i in range(self.n):
                self.trade_network.append(list(range(i+1, self.n))+list(range(i)))
        else:
            raise ValueError('Unacceptable vote trading network topology')         
            
# Trade votes
    def trade_votes(self):
        self.create_network()
        self.votes_owned = [[[i] for _ in range(self.m)] for i in range(self.n)]

        stuck = False
        stuck_counter = 0
        indx = 0
        self.trade_count = 0
        
        while not stuck:
            trade=False
            trade_pairs = []
            for i in range(1,self.m):
                for j in range(i):
                    if 0<len(self.votes_owned[indx][i])<self.n/2 and 0<len(self.votes_owned[indx][j])<self.n/2:
                        if np.abs(self.utilities[indx][i])-np.abs(self.utilities[indx][j]) >= self.trade_thresh:
                            trade_pairs.append([i,j])
                        elif np.abs(self.utilities[indx][j])-np.abs(self.utilities[indx][i]) >= self.trade_thresh:
                            trade_pairs.append([j,i])
            for nindx in self.trade_network[indx]:
                potential_trades = []
                for pair in trade_pairs:
                    i = pair[0]
                    j = pair[1]
                    if 0<len(self.votes_owned[nindx][i])<self.n/2 and 0<len(self.votes_owned[nindx][j])<self.n/2:
                        if np.abs(self.utilities[nindx][j])-np.abs(self.utilities[nindx][i])>=self.trade_thresh:
                            potential_trades.append(pair)
                if potential_trades:
                    trade=True
                    break
            
            if trade==True:
                trade_pair = rand.choice(potential_trades)
                i = trade_pair[0]
                j = trade_pair[1]
                if self.votes_owned[indx][j]==[indx]:
                    indx_give = indx
                else:
                    indx_give = rand.choice([k for k in self.votes_owned[indx][j] if k!=indx])
                if self.votes_owned[nindx][i]==[nindx]:
                    nindx_give = nindx
                else:
                    nindx_give = rand.choice([i for i in self.votes_owned[nindx][i]])
                
                self.votes_owned[indx][i].append(nindx_give)
                self.votes_owned[indx][j].remove(indx_give)
                self.votes_owned[nindx][j].append(indx_give)
                self.votes_owned[nindx][i].remove(nindx_give)
                
                stuck_counter=0
                self.trade_count+=1
                
            else:
                stuck_counter += 1
                
            if stuck_counter == self.n:
                stuck=True
                
            indx = (indx+1)%self.n
        
        self.traded_votes = []
        for indx in range(self.n):
            indx_votes = []
            for t in range(self.m):
                nindx = [i for i in range(self.n) if indx in self.votes_owned[i][t]][0]
                if self.utilities[nindx][t]>0:
                    indx_votes.append(1)
                else:
                    indx_votes.append(-1)
            self.traded_votes.append(indx_votes)
        
# Trade votes, then vote like majority rule
    def vote_post_trades(self):
        self.trade_votes()
        
        self.vt_value = 0
        self.vt_sup_issues = 0
        self.vt_maj_agreement = 0
        
        for i in range(self.m):
            if sum([self.traded_votes[j][i] for j in range(self.n)])>0:
                self.vt_value += self.util_value[i]
                self.vt_sup_issues += 1
                if self.voter_support[i]>self.n/2:
                    self.vt_maj_agreement += 1
            elif sum([self.traded_votes[j][i] for j in range(self.n)])<0:
                self.vt_value -= self.util_value[i]
                if self.voter_support[i]<self.n/2:
                    self.vt_maj_agreement += 1
            else:
                print('error with traded votes')

    def split_bundle(self):
        self.bundles = [[] for _ in range(self.bundle_num)]
        if self.bundle_split=='random':
            for i in range(self.m):
                rand.choice(self.bundles).append(i)
        elif self.bundle_split=='balanced_random':
            for i in range(self.m):
                self.bundles[i%self.bundle_num].append(i)
        elif self.bundle_split=='balanced_odds_only':
            for i in range(self.m):
                self.bundles[i%self.bundle_num].append(i)
            bundle_sizes = [len(x) for x in self.bundles]
            indx = -1
            for i in range(self.bundle_num):
                if bundle_sizes[i]%2==0:
                    if indx==-1:
                        indx = i
                    else:
                        nindx = i
                        self.bundles[indx].append(self.bundles[nindx].pop(-1))
                        indx = -1
        elif self.bundle_split=='maj-min':
            for i in range(self.m):
                if self.voter_support[i]>self.n/2:
                    self.bundles[0].append(i)
                else:
                    self.bundles[1].append(i)
        elif self.bundle_split=='left-right' and self.util_method=='spatial':
            for i in range(self.m):
                if self.issue_positions[i][0][0]<self.issue_positions[i][1][0]:
                    self.bundles[0].append(i)
                else:
                    self.bundles[1].append(i)
        elif self.bundle_split=='voter_copy':
            for i in range(self.m):
                if self.utilities[0][i]>0:
                    self.bundles[0].append(i)
                else:
                    self.bundles[1].append(i)

        else:
            print('Unacceptable Split Method')

        self.split_bundle_value = 0
        self.split_bundle_sup_issues = 0
        self.split_bundle_maj_agreement = 0
        self.split_bundle_vi_pair_agreement = 0

        # 1 = pass, -1 = fail, 0 = tie
        self.bundle_passage = []

        for bundle in self.bundles:
            subbundle_votes = 0
            subbundle_value = 0
            for i in range(self.n):
                total_util = 0
                for j in bundle:
                    total_util += self.utilities[i][j]
                subbundle_value += total_util
                if total_util>0:
                    subbundle_votes += 1
                elif total_util==0:
                    subbundle_votes += 1/2
                # elif total_util==0 and self.utilities[i][0]>0:
                #     subbundle_votes += 1
                
                # if i==0:
                #     if total_util>0 or total_util==0 and self.utilities[0][0]>0:
                #         chair_support = True
                #     else:
                #         chair_support = False
            
            if subbundle_votes > self.n/2:
                self.split_bundle_sup_issues += len(bundle)
                self.split_bundle_value += subbundle_value
                self.bundle_passage.append(1)
                for i in bundle:
                    self.split_bundle_vi_pair_agreement += self.voter_support[i]
                    if self.voter_support[i]>self.n/2:
                        self.split_bundle_maj_agreement+=1
                        
            elif subbundle_votes < self.n/2:
                self.split_bundle_value -= subbundle_value
                self.bundle_passage.append(-1)
                for i in bundle:
                    self.split_bundle_vi_pair_agreement += self.n-self.voter_support[i]
                    if self.voter_support[i]<self.n/2:
                        self.split_bundle_maj_agreement+=1
                        
            else:
                self.bundle_passage.append(0)
                for i in bundle:
                    self.split_bundle_vi_pair_agreement += self.n/2
                    self.split_bundle_maj_agreement+=0.5
            
            # elif subbundle_votes()==self.n/2 and chair_support:
            #     self.split_bundle_sup_issues += len(bundle)
            #     self.split_bundle_value += subbundle_value
            #     self.bundle_passage.append(1)
            #     for i in bundle:
            #         self.split_bundle_vi_pair_agreement += self.voter_support[i]
            #         if self.voter_support[i]>self.n/2:
            #             self.split_bundle_maj_agreement+=1            
            # else:
            #     self.split_bundle_value -= subbundle_value
            #     self.bundle_passage.append(-1)
            #     for i in bundle:
            #         self.split_bundle_vi_pair_agreement += self.n-self.voter_support[i]
            #         if self.voter_support[i]<self.n/2:
            #             self.split_bundle_maj_agreement+=1
                    
        # Determine net gain for those who wanted to split and those who did not
        self.splitter_gain = 0
        self.nonsplitter_gain = 0
        self.splitter_count = 0
        
        self.nonsplit_pass = 0
        self.nonsplit_fail = 0

        for i in range(self.n):
            passage = False
            failure = False
            
            if self.bundle_pass:
                old_val = sum([self.utilities[i][j] for j in range(self.m)])
            else:
                old_val = sum([-self.utilities[i][j] for j in range(self.m)])
            
            new_val = 0
            for k in range(self.bundle_num):
                temp = sum(self.utilities[i][j] for j in self.bundles[k])
                if temp>0:
                    passage=True
                elif temp<0:
                    failure=True
                
                if self.bundle_passage[k]==1:
                    new_val += temp
                elif self.bundle_passage[k]==-1:
                    new_val -= temp
                    
            if passage and failure:
                self.splitter_count += 1
                self.splitter_gain += (new_val - old_val)
            else:
                self.nonsplitter_gain += (new_val - old_val)
                
            if passage and not failure:
                self.nonsplit_pass += 1
                
            if not passage and failure:
                self.nonsplit_fail += 1
            
        if self.splitter_count != 0:    
            self.splitter_gain = self.splitter_gain/self.splitter_count
        if self.splitter_count != self.n:
            self.nonsplitter_gain = self.nonsplitter_gain/(self.n - self.splitter_count)
            
# Measure the gerrymanderability of a voting preference profile
    def gerry_measure_odds_only(self):
        self.subsets = []
        self.subset_votes = []
        self.mean_median = []
        self.eg = []
        self.nonmaj_frac = []
        self.subset_sizes = list(range(1,self.m+1, 2))
        issues = list(range(self.m))
        
        for size in self.subset_sizes:
            subsets_of_size = list(combinations(issues, size))
            for subset in subsets_of_size:
                subbundle = list(subset)
                self.subsets.append(subbundle)
                yes_votes = []
                for issue in subbundle:
                    yes_votes.append(self.voter_support[issue])
                # self.mean_median.append(np.abs(np.mean(yes_votes)-np.median(yes_votes)))
                non_maj_issues = 0
                subbundle_votes = 0 
                for i in range(self.n):
                    ind_util = 0
                    for s in subbundle:
                        ind_util += self.utilities[i][s]
                    if ind_util > 0:
                        subbundle_votes += 1
                    ###################
                    # added for real data section
                    elif ind_util == 0:
                        subbundle_votes += 0.5
                    ###################
                self.subset_votes.append(subbundle_votes)
                for issue in subbundle:
                    if (subbundle_votes > self.n/2 and self.voter_support[issue] < self.n/2) or (subbundle_votes < self.n/2 and self.voter_support[issue] > self.n/2):
                        non_maj_issues += 1
                self.nonmaj_frac.append(non_maj_issues/size)
                
                # compute mean-median score
                ind_yes_votes = []
                for i in range(self.n):
                    yes_vote_count = 0
                    for issue in subbundle:
                        if self.utilities[i][issue]>0:
                            yes_vote_count += 1
                    ind_yes_votes.append(yes_vote_count)
                self.mean_median.append(np.abs(np.mean(ind_yes_votes)-np.median(ind_yes_votes))/size)
                # compute efficiency gap
                yes_wasted = 0
                no_wasted = 0
                for c in ind_yes_votes:
                    if c>size/2:
                        yes_wasted += (c-(int(size/2)+1))
                        no_wasted += size-c
                    else:
                        yes_wasted += c
                        no_wasted += (int(size/2))-c
                self.eg.append(np.abs(yes_wasted-no_wasted)/(self.n*size))

    def gerry_measure_all_subsets(self):
        self.subsets = []
        self.subset_votes = []
        self.mean_median = []
        self.eg = []
        self.nonmaj_frac = []
        self.subset_sizes = list(range(1,self.m+1))
        issues = list(range(self.m))
        
        for size in self.subset_sizes:
            subsets_of_size = list(combinations(issues, size))
            for subset in subsets_of_size:
                subbundle = list(subset)
                self.subsets.append(subbundle)
                yes_votes = []
                for issue in subbundle:
                    yes_votes.append(self.voter_support[issue])
                # self.mean_median.append(np.abs(np.mean(yes_votes)-np.median(yes_votes)))
                non_maj_issues = 0
                subbundle_votes = 0 
                for i in range(self.n):
                    ind_util = 0
                    for s in subbundle:
                        ind_util += self.utilities[i][s]
                    if ind_util > 0:
                        subbundle_votes += 1
                    ###################
                    # added for real data section
                    elif ind_util == 0:
                        subbundle_votes += 0.5
                    ###################
                self.subset_votes.append(subbundle_votes)
                for issue in subbundle:
                    if (subbundle_votes > self.n/2 and self.voter_support[issue] < self.n/2) or (subbundle_votes < self.n/2 and self.voter_support[issue] > self.n/2):
                        non_maj_issues += 1
                self.nonmaj_frac.append(non_maj_issues/size)
                
                # compute mean-median score
                ind_yes_votes = []
                for i in range(self.n):
                    yes_vote_count = 0
                    for issue in subbundle:
                        if self.utilities[i][issue]>0:
                            yes_vote_count += 1
                    ind_yes_votes.append(yes_vote_count)
                self.mean_median.append(np.abs(np.mean(ind_yes_votes)-np.median(ind_yes_votes))/size)
                # compute efficiency gap
                yes_wasted = 0
                no_wasted = 0
                for c in ind_yes_votes:
                    if c>size/2:
                        yes_wasted += (c-(int(size/2)+1))
                        no_wasted += size-c
                    else:
                        yes_wasted += c
                        no_wasted += (int(size/2))-c
                self.eg.append(np.abs(yes_wasted-no_wasted)/(self.n*size))



# Compute the utility for each voter from the bundled vote
    def ind_util_bundle(self):
        utils = []
        if self.bundle_pass:
            for i in range(self.n):
                util = sum(self.utilities[i])
                utils.append(util)
        else:
            for i in range(self.n):
                util = -1*sum(self.utilities[i])
                utils.append(util)
        return utils
            
    
    def ind_util_issuebyissue(self):
        utils = []
        for i in range(self.n):
            util = 0
            for j in range(self.m):
                if self.maj_rule_passage[j]:
                    util += self.utilities[i][j]
                else:
                    util -= self.utilities[i][j]
            utils.append(util)
        return utils
        

#####################
# Misc functions
#####################

    def dist(self,a,b):
        val = 0
        for i in range(len(a)):
            val += (a[i]-b[i])**2
        return np.sqrt(val)     
       
    def print_all_info(self):
        print(f'Number of voters is {self.n}')
        print(f'Number of issues is {self.m}')
        print(f'Utilities derived from {self.util_method} model')
        print()
        
        print(f'Maximum utility is {self.util_rule_value}')
        print(f'Utility of bundled vote is {self.bundle_value}')
        print(f'Utility of majority rule is {self.maj_rule_value}')
        print(f'Utility of vote trading is {self.vt_value}')
        print()
        
        print(f'Majority rule supports {self.maj_rule_sup_issues}')
        print(f'Utility rule supports {self.util_rule_sup_issues}')
        print(f'Utility rule agrees with majority rule on {self.util_maj_agreement} issues')
        print(f'Bundling supports {self.bundle_sup_issues}')
        print(f'Bundling agrees with majority rule on {self.bundle_maj_agreement} issues')
        print(f'Vote trading supports {self.vt_sup_issues}')
        print(f'Vote trading agrees with majority rule on {self.vt_maj_agreement} issues')
        print()
        
        print(f'Vote trading found {self.trade_count} trades')
            
            
            
            
            
            
            
            
            
            
            
            