  
"""
The function fastNonDominatedSort is based on the sorting algorithm described by
Deb, Kalyanmoy, et al.
"A fast and elitist multiobjective genetic algorithm: NSGA-II."
IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
"""

import numpy as np
import pdb

from functions_hv_python3 import HyperVolume


def crowding_distance_sorting(fitness_values):
    num_objectives, num_individuals = fitness_values.shape
    sorted_fitness_indices = np.argsort(fitness_values, axis=1)
    crowding_distance = np.zeros(num_individuals)

    for i in range(num_objectives):
        sorted_fitness_values = fitness_values[i][sorted_fitness_indices[i]]
        crowding_distance[sorted_fitness_indices[i][0]] = np.inf
        crowding_distance[sorted_fitness_indices[i][-1]] = np.inf
        for j in range(1, num_individuals - 1):
            crowding_distance[sorted_fitness_indices[i][j]] += (
                    sorted_fitness_values[j + 1] - sorted_fitness_values[j - 1])

    return crowding_distance


def select_best_individuals(fitness_values, n):
    num_objectives, num_individuals = fitness_values.shape
    crowding_distance = crowding_distance_sorting(fitness_values)
    sorted_indices = np.argsort(crowding_distance)[::-1]
    best_individuals_indices = sorted_indices[:n]
    return best_individuals_indices

def determine_non_dom_mo_sol(mo_obj_val):
    # get set of non-dominated solutions, returns indices of non-dominated and booleans of dominated mo_sol
    n_mo_sol = mo_obj_val.shape[1]
    domination_rank = fastNonDominatedSort(mo_obj_val)
    non_dom_indices = np.where(domination_rank == 0)
    non_dom_indices = non_dom_indices[0] # np.where returns a tuple, so we need to get the array inside the tuple
    # non_dom_mo_sol = mo_sol[:,non_dom_indices]
    # non_dom_mo_obj_val = mo_obj_val[:,non_dom_indices]
    mo_sol_is_non_dominated = np.zeros(n_mo_sol,dtype = bool)
    mo_sol_is_non_dominated[non_dom_indices] = True
    mo_sol_is_dominated = np.bitwise_not(mo_sol_is_non_dominated)
    return(non_dom_indices,mo_sol_is_dominated)

def fastNonDominatedSort(objVal):
    # Based on Deb et al. (2002) NSGA-II
    N_OBJECTIVES = objVal.shape[0] 
    N_SOLUTIONS = objVal.shape[1]

    rankIndArray = - 999 * np.ones(N_SOLUTIONS, dtype = int) # -999 indicates unassigned rank
    solIndices = np.arange(0,N_SOLUTIONS) # array of 0 1 2 ... N_SOLUTIONS
    ## compute the entire domination matrix
    # dominationMatrix: (i,j) is True if solution i dominates solution j
    dominationMatrix = np.zeros((N_SOLUTIONS,N_SOLUTIONS), dtype = bool)
    for p in solIndices:
        objValA = objVal[:,p][:,None] # add [:,None] to preserve dimensions
        # objValArray =  np.delete(objVal, obj = p axis = 1) # dont delete solution p because it messes up indices
        dominates = checkDomination(objValA,objVal)
        dominationMatrix[p,:] = dominates

    # count the number of times a solution is dominated
    dominationCounter = np.sum(dominationMatrix, axis = 0)

    ## find rank 0 solutions to initialize loop
    isRankZero = (dominationCounter == 0) # column and row binary indices of solutions that are rank 0

    rankZeroRowInd = solIndices[isRankZero] 
    # mark rank 0's solutions by -99 so that they are not considered as members of next rank
    dominationCounter[rankZeroRowInd] = -99
    # initialize rank counter at 0
    rankCounter = 0
    # assign solutions in rank 0 rankIndArray = 0
    rankIndArray[isRankZero] = rankCounter

    isInCurRank = isRankZero
    # while the current rank is not empty
    while not (np.sum(isInCurRank) == 0):
        curRankRowInd = solIndices[isInCurRank] # column and row numbers of solutions that are in current rank 
        # for each solution in current rank
        for p in curRankRowInd:
            # decrease domination counter of each solution dominated by solution p which is in the current rank
            dominationCounter[dominationMatrix[p,:]] -= 1 #dominationMatrix[p,:] contains indices of the solutions dominated by p
        # all solutions that now have dominationCounter == 0, are in the next rank		
        isInNextRank = (dominationCounter == 0)
        rankIndArray[isInNextRank] = rankCounter + 1	
        # mark next rank's solutions by -99 so that they are not considered as members of future ranks
        dominationCounter[isInNextRank] = -99
        # increase front counter
        rankCounter += 1
        # check which solutions are in current rank (next rank became current rank)
        isInCurRank = (rankIndArray == rankCounter)
        if not np.all(isInNextRank == isInCurRank): # DEBUGGING, if it works fine, replace above assignment
            pdb.set_trace()
    return(rankIndArray)

def checkDomination(objValA,objValArray):
    dominates = ( np.any(objValA < objValArray, axis = 0) & np.all(objValA <= objValArray , axis = 0) )
    return(dominates)

def compute_hv_in_higher_dimensions(mo_obj_val,ref_point):
    n_mo_obj = mo_obj_val.shape[0]
    n_mo_sol = mo_obj_val.shape[1]
    assert len(ref_point) == n_mo_obj
    # initialize hv computation instance
    hv_computation_instance = HyperVolume(tuple(ref_point))
    # turn numpy array to list of tuples
    list_of_mo_obj_val = list()
    for i_mo_sol in range(n_mo_sol):
        list_of_mo_obj_val.append(tuple(mo_obj_val[:,i_mo_sol]))

    hv = float(hv_computation_instance.compute(list_of_mo_obj_val))
    return(hv)


import numpy as np

def compare(p1, p2):
    # return 0同层 1 p1支配p2  -1 p2支配p1
    D = len(p1)
    p1_dominate_p2 = True  # p1 更小
    p2_dominate_p1 = True
    for i in range(D):
        if p1[i] > p2[i]:
            p1_dominate_p2 = False
        if p1[i] < p2[i]:
            p2_dominate_p1 = False

    if p1_dominate_p2 == p2_dominate_p1:
        return 0
    return 1 if p1_dominate_p2 else -1

def fast_non_dominated_sort(P):
    P_size = len(P)
    n = np.full(shape=P_size, fill_value=0)    # 被支配数
    S = []    # 支配的成员
    f = []    # 0 开始每层包含的成员编号们
    rank = np.full(shape=P_size, fill_value=-1)  # 所处等级

    f_0 = []
    for p in range(P_size):
        n_p = 0
        S_p = []
        for q in range(P_size):
            if p == q:
                continue
            cmp = compare(P[p], P[q])
            if cmp == 1:
                S_p.append(q)
            elif cmp == -1:  # 被支配
                n_p += 1
        S.append(S_p)
        n[p] = n_p
        if n_p == 0:
            rank[p] = 0
            f_0.append(p)
    f.append(f_0)

    i = 0
    while len(f[i]) != 0:  # 可能还有i+1层
        Q = []
        for p in f[i]:  # i层中每个个体
            for q in S[p]:  # 被p支配的个体
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        f.append(Q)
    rank +=1
    return rank,f




