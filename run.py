"""
Runing the proposed Paret Set Learning (PSL) method on 8 test problems.
"""
import copy

import numpy as np
import torch
from functions_evaluation import fast_non_dominated_sort, crowding_distance_sorting, select_best_individuals
from problem import get_problem
from pymoo.indicators.hv import HV
from model import ParetoSetModel
from matplotlib import pyplot as plt
import arviz as az
from pymoo.core.mating import Mating
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
import random
import argparse
from pymoo.operators.selection.rnd import RandomSelection
import os
from pymoo.operators.selection.tournament import compare, TournamentSelection
from tqdm import tqdm
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.individual import Individual
def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)

def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

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

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem_name', '-pn', default='dtlz5', type=str, help='Test problems.')
    parser.add_argument('--scalar', default='hv2', type=str)
    parser.add_argument('--N', default=20, type=int)
    parser.add_argument('--using_adapt', default=0, type=int)
    parser.add_argument('--calpop', default=0, type=int)
    parser.add_argument('--crossprob', default=0.9, type=float)
    parser.add_argument('--mutationprob', default=0.9, type=float)
    args = parser.parse_args()
    # list of 15 test problems, which are defined in problem.py
    ins_list = ['f1','f2','f3','f4','f5','f6',
                'vlmop1','vlmop2', 'vlmop3', 'dtlz2',
                're21', 're23', 're33','re36','re37']

    set_seed(2)
    ins_list = [args.problem_name]
    problem_dim = {'zdt3': 10, 'dtlz2':10, 'dtlz5':10, 'dtlz7':10, 're21':4, 're31':3, 're36':4,
                   're37':4, 're33':4, 're34':5}
    n_dim = problem_dim[args.problem_name]
    # number of independent runs
    n_run = 11 #20
    n_sample = 5
    # PSL
    # number of learning steps
    n_steps = 1001
    # number of sampled preferences per step
    n_pref_update = 8
    # Subset seletion
    if args.problem_name in ['dtlz7', 're21', 're36', 'dtlz5']:
        seed = [4, 6, 9, 10, 11] # for dtlz7 and others
    elif args.problem_name in ['dtlz4', 're37', 're31']:
        seed = [4, 6, 9, 11, 12] # for dtlz4 re37 re31
    else:
        seed = [4, 6, 16, 10, 11] # for zdt3
    # device
    device = 'cuda'
    # -----------------------------------------------------------------------------
    pop_hv_list = []
    hv_list = []
    print('args:', args)
    for test_ins in ins_list:
        print(test_ins)
        all_offspring = []
        # get problem info
        # if args.problem_name in ['dtlz4', 'zdt3', 'dtlz5', 'dtlz7']:
        #     problem = get_problem(test_ins, n_dim=n_dim)
        # else:
        problem = get_problem(test_ins, n_dim=n_dim)
        n_dim = problem.n_dim
        n_obj = problem.n_obj

        ref_point = problem.nadir_point
        ref_point = [1.1*x for x in ref_point]
        mutation = PolynomialMutation(prob= args.mutationprob,eta=20)
        cross_over = SBX(prob=args.crossprob, eta=15)
        eliminate_duplicates = DefaultDuplicateElimination()
        # repeatedly run the algorithm n_run times
        for run_iter in range(n_run):
            record = 0
            record_pop = 0
            s_hv_list = []
            s_hv_list_pop = []
            print(f'######################iter{run_iter}#####################')
            # TODO: better I/O between torch and np
            # currently, the Pareto Set Model is on torch, and the Gaussian Process Model is on np
            z = torch.tensor([]).to(device) + torch.inf
            psmodel = ParetoSetModel(n_dim, n_obj, problem)
            psmodel.to(device)

            # optimizer
            optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
            mcmc = 0
            # t_step Pareto Set Learning with Gaussian Process
            pref_archive = []
            archive = []
            angles = []
            for t_step in tqdm(range(n_steps), desc='Test'):
                psmodel.train()
                # sample n_pref_update preferences
                if mcmc and args.using_adapt:
                    population = []
                    if len(used_data) == 1:
                        pref_vec = np.random.dirichlet(alpha,n_pref_update)
                    else:
                        for _ in range(n_pref_update):
                            p = np.array(angle_archive) / sum(np.array(angle_archive))

                            # index = np.random.choice(range(len(used_data)), p=p.ravel(), replace=False, size=2)
                            samples = random.sample(used_data, 2)
                            population.append([Individual(X=np.array(samples[0])), Individual(X=np.array(samples[1]))])
                        pref_problem = problem
                        pref_problem.n_var = n_obj
                        pref_problem.xl = np.array([0]*n_obj)
                        pref_problem.xu = np.array([1]*n_obj)
                        off = cross_over.do(pref_problem, population)
                        off = mutation.do(pref_problem, off)
                        off = eliminate_duplicates.do(off).get("X")
                        off_X = off / off.sum(1).reshape(-1, 1)
                        pref_vec = []
                        for i in range(n_pref_update):
                            pref_vec.append(off_X[i])

                    pref_vec = torch.tensor(np.array(pref_vec)).float()+ 0.0001
                    pref_vec = pref_vec.cuda()
                else:
                    alpha = np.ones(n_obj) * 1
                    pref = np.random.dirichlet(alpha,n_pref_update)
                    pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
                    pref_vec.to(device).float()
                    # sampling process of mixture distribution
                all_offspring.append(pref_vec.cpu().numpy().tolist())
                # get the current coressponding solutions
                x = psmodel(pref_vec)
                value = problem.evaluate(x)

                if t_step==0:

                    z = torch.min(torch.cat([value.to(z.dtype), z]), axis=0).values.data
                    yis = torch.tensor([0.9]*n_obj).to(x.device)
                    yis[z<0] = 1.1

                    max_ = torch.max(value-z, 0).values.data
                    min_ = torch.min(value-z, 0).values.data

                    if args.problem_name in ['re21', 're31', 're33', 're36', 'zdt3', 'dtlz5']:
                        value_ = (value-yis*z) / (max_-min_+0.00001)
                        rref4hv = (torch.tensor(ref_point).cuda() - yis * z) / (max_ - min_ + 0.00001)
                    else:
                        value_ = (value - yis * z)
                        rref4hv = (torch.tensor(ref_point).cuda() - yis * z)
                else:
                    z = torch.min(torch.cat([value.to(z.dtype), z.reshape(1, n_obj)]), axis=0).values.data
                    yis = torch.tensor([0.9]*n_obj).to(x.device)
                    yis[z<0] = 1.1
                    max_ = torch.max(value-z, 0).values.data
                    min_ = torch.min(value-z, 0).values.data
                    if args.problem_name in ['re21', 're31', 're33', 're36', 'zdt3', 'dtlz5']:
                        value_ = (value-yis*z)/ (max_-min_+0.00001)
                        rref4hv =  (torch.tensor(ref_point).cuda()-yis*z)/ (max_-min_+0.00001)
                    else:
                        value_ = (value - yis * z)
                        rref4hv = (torch.tensor(ref_point).cuda() - yis * z)
                angles.extend(torch.cosine_similarity(value_, 1/pref_vec).detach().cpu().numpy().tolist())
                archive.extend(value.cpu().detach().numpy().tolist())
                pref_archive.extend(pref_vec.cpu().detach().numpy().tolist())
                if args.scalar == 'tch2':
                    tch_value = torch.max(pref_vec * value_, axis=1)[0]
                    loss = torch.sum(tch_value)
                elif args.scalar == 'tch1':
                    pref_vec = (1/pref_vec) / torch.sum(1/pref_vec, dim=1, keepdim=True)
                    tch_value = torch.max(pref_vec * value_, axis=1)[0]
                    loss = torch.sum(tch_value)
                elif args.scalar == 'cosmos':
                    loss = torch.sum(pref_vec * value_, dim=1).mean()
                    penalty = (torch.sum(value_ * pref_vec, dim=1) /
                               (torch.norm(value_, dim=1) * torch.norm(pref_vec, dim=1))).mean()
                    loss -= 100 * penalty
                elif args.scalar == 'hv1':
                    if n_obj == 2:
                        pref = np.stack([np.linspace(0, 1, 101), 1 - np.linspace(0, 1, 101)]).T
                        test_pref = torch.tensor(pref).to(device).float()
                    elif n_obj == 3:
                        test_pref = torch.tensor(das_dennis(20, 3)).to(device).float()  # 105
                    value_ = rref4hv - value_
                    tch_loss = torch.zeros(len(test_pref)).cuda()
                    for v in test_pref:
                        tch_loss += torch.max(torch.min((1 / (v + 0.0001)) * value_, axis=1)[0])

                    tch_loss[tch_loss > 0] = tch_loss[tch_loss > 0]**n_obj
                    loss = -torch.sum(tch_loss)
                else:
                    loss = torch.sum(pref_vec * value_, dim=1).sum()
                # gradient-based pareto set model update
                optimizer.zero_grad()
                loss.backward()
                #psmodel(pref_vec).backward(tch_grad)
                optimizer.step()
                if args.using_adapt:
                    if (t_step+1) % 50 == 0:
                        used_data = []
                        use_archive = []
                        angle_archive = []
                        rank_re, front_re = fast_non_dominated_sort(np.array(archive))
                        if len(front_re[0]) < args.N:
                            for idx in front_re[0]:
                                use_archive.append(archive[idx])
                                used_data.append(pref_archive[idx])
                                angle_archive.append(angles[idx])
                        else:
                            selected_idx = select_best_individuals(np.array(archive).T,
                                                                   args.N)
                            use_archive = [archive[idx] for idx in selected_idx]
                            used_data = [pref_archive[idx] for idx in selected_idx]
                            angle_archive = [angles[idx] for idx in selected_idx]
                        archive = []
                        print('-------', len(used_data))
                        pref_archive = []
                        angles = []
                        mcmc = 1

                psmodel.eval()
                # if (t_step + 1) % 50 == 0:
                with torch.no_grad():

                    generated_pf = []
                    generated_ps = []


                    if n_obj == 2:
                        pref = np.stack([np.linspace(0,1,248), 1 - np.linspace(0,1,248)]).T
                        pref = torch.tensor(pref).to(device).float()

                    if n_obj == 3:

                        pref_size = 105
                        pref = torch.tensor(das_dennis(30,3)).to(device).float()  # 105

                    sol = psmodel(pref)
                    obj = problem.evaluate(sol)
                    generated_ps = sol.cpu().numpy()
                    generated_pf = obj.cpu().numpy()
                    hv = HV(ref_point=ref_point)
                    hv_value = hv(generated_pf)
                    if record and hv_value < s_hv_list[t_step-1]:
                        hv_value = s_hv_list[t_step-1]
                    # else:
                    #     np.save(f'pf/{args.scalar}_{args.problem_name}_{args.using_adapt}_paretofront_{run_iter}.npy',
                    #             generated_pf)
                    record = 1
                    s_hv_list.append(hv_value)
                    print("hv", "{:.2e}".format(hv_value))
                    if args.calpop and args.using_adapt and (t_step+1)>51:
                        if len(used_data) == 1:
                            s_hv_list_pop.append(0)
                        else:
                            population2 = []
                            seq = 0
                            for _ in range(len(pref)*2):
                                samples2 = random.sample(used_data, 2)
                                population2.append([Individual(X=np.array(samples2[0])), Individual(X=np.array(samples2[1]))])
                            pref_problem2 = copy.deepcopy(problem)
                            pref_problem2.n_var = n_obj
                            pref_problem2.xl = np.array([0]*n_obj)
                            pref_problem2.xu = np.array([1]*n_obj)
                            off2 = cross_over.do(pref_problem2, population2)
                            off2 = mutation.do(pref_problem2, off2)
                            off2 = eliminate_duplicates.do(off2).get("X")
                            off_X2 = off2 / off2.sum(1).reshape(-1, 1)
                            pref_vec2 = []
                            for i in range(len(pref)):
                                pref_vec2.append(off_X2[i])
                            pref_vec2 = torch.tensor(pref_vec2).to(device).float()
                            sol2 = psmodel(pref_vec2)
                            obj2 = problem.evaluate(sol2)
                            generated_ps_pop = sol2.cpu().numpy()
                            generated_pf_pop = obj2.cpu().numpy()
                            hv_value_pop = hv(generated_pf_pop)
                            if record_pop and hv_value_pop < s_hv_list_pop[t_step - 52]:
                                hv_value_pop = s_hv_list_pop[t_step - 52]

                            s_hv_list_pop.append(hv_value_pop)
                            print("pop hv", "{:.2e}".format(hv_value_pop))
                        record_pop = 1

            if args.calpop:
                pop_hv_list.append(s_hv_list_pop)
            hv_list.append(s_hv_list)

    np.save(f'pf/{args.scalar}_{args.problem_name}_{args.using_adapt}_N{args.N}_cp_{args.crossprob}_mp_{args.mutationprob}.npy', np.array(hv_list))
    if args.calpop and args.using_adapt:
        np.save(f'pf/{args.scalar}_{args.problem_name}_{args.using_adapt}_N{args.N}_POP.npy', np.array(pop_hv_list))
    # np.save(f'pf/{args.scalar}_{args.problem_name}_{args.using_adapt}_finally_pop.npy', np.array(used_data))
    # np.save(f'pf/{args.scalar}_{args.problem_name}_{args.using_adapt}_all_offspring.npy', np.array(all_offspring))
    print('args:', args)
    print(np.array(hv_list).max(1))
                # with open('hv_psl_mobo.pickle', 'wb') as output_file:
                #     pickle.dump([hv_list], output_file)

