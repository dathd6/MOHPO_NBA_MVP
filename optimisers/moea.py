import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from models.mvp_prediction_model import MVPPredictionModel

# Constant variables & Utils function
from constants import LEARNING_RATE, \
                      N_ESTIMATORS, \
                      MUTATION_RATE

from utils import generate_random_number

class MOEA:
    def __init__(self, nba_stats, pop_size, n_evaluations, mvp_share_rmse=False, execution_time=False, mvp_ranking_rmse=False, population=[]) -> None:
        """Initialise Multi-objective attribute"""
        self.eval_count = 0
        self.size_p = pop_size
        self.n_evaluations = n_evaluations
        self.objectives = np.array([mvp_share_rmse, execution_time, mvp_ranking_rmse])
        self.nba_stats = nba_stats
        if len(population) != 0:
            self.population = population
        else:
            self.population = np.array([MVPPredictionModel(
                                            nba_stats,
                                            self.objectives
                                        ) for _ in range(self.size_p)])
        self.n_objectives = len(self.objectives[self.objectives == True])
        self.perf_metrics = [] # Hypervolume metrics through generations
        self.fronts = []

        ref_point = self.get_nadir()
        self.ind = HV(ref_point=ref_point + 0.5)

    def get_nadir(self):
        fitnesses = np.array([solution.get_fitness() for solution in self.population])
        nadir = []
        for i in range(self.n_objectives):
            nadir.append(fitnesses[:, i].max())
        return np.array(nadir)

    def non_dominated_sorting(self):
        """Fast non-dominated sorting to get list Pareto Fronts"""
        dominating_sets = []
        dominated_counts = []

        # For each solution:
        # - Get solution index that dominated by current solution
        # - Count number of solution dominated current solution
        for solution_1 in self.population:
            current_dominating_set = set()
            dominated_counts.append(0)
            for i, solution_2 in enumerate(self.population):
                if solution_1 >= solution_2 and not solution_1 == solution_2:
                    current_dominating_set.add(i)
                elif solution_2 >= solution_1 and not solution_2 == solution_1:
                    dominated_counts[-1] += 1
            dominating_sets.append(current_dominating_set)

        dominated_counts = np.array(dominated_counts)
        self.fronts = []

        # Append all the pareto fronts and stop when there is no solution being dominated (domintead count = 0)
        while True:
            current_front = np.where(dominated_counts==0)[0]
            if len(current_front) == 0:
                break
            self.fronts.append(current_front)
            for individual in current_front:
                dominated_counts[individual] = -1 # this solution is already accounted for, make it -1 so will not find it anymore
                dominated_by_current_set = dominating_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    dominated_counts[dominated_by_current] -= 1

    def calc_performance_metric(self):
        """Calculate hypervolume to the reference point"""
        front = self.fronts[0]
        solutions = np.array([solution.get_fitness() for solution in self.population[front]])
        self.perf_metrics.append(
            [self.eval_count, self.ind(solutions)]
        )

    def visualize_objective_space(self, title, figsize, labels, c_dominated='#2377B4', c_non_dominated='#FA8010', alpha=1, view_init=None):
        front = self.fronts[0]
        non_dominated = np.array([
            solution.get_fitness() for solution in self.population[front]
        ])
        dominated = []
        for i in range(1, len(self.fronts)):
            dominated = dominated + [solution.get_fitness() for solution in self.population[self.fronts[i]]]
        dominated = np.array(dominated)
        ax = None
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        if self.n_objectives == 2:
            ax = fig.add_subplot(121)
            if dominated.size != 0:
                sns.scatterplot(
                    x=dominated[:, 0],
                    y=dominated[:, 1],
                    ax=ax,
                    label='dominated',
                    color=c_dominated,
                    alpha=alpha
                )
            sns.scatterplot(
                x=non_dominated[:, 0],
                y=non_dominated[:, 1],
                ax=ax,
                color=c_non_dominated,
                label='non-dominated',
            )

        elif self.n_objectives == 3:
            ax = fig.add_subplot(121, projection='3d')
            if dominated.size != 0:
                ax.scatter(
                    dominated[:, 0],
                    dominated[:, 1],
                    dominated[:, 2],
                    label='dominated',
                    color=c_dominated,
                    alpha=alpha
                )
            ax.scatter(
                non_dominated[:, 0],
                non_dominated[:, 1],
                non_dominated[:, 2],
                color=c_non_dominated,
                label='non-dominated'
            )
            ax.set_zlabel(labels[2])
            ax.view_init(*view_init)
            plt.legend()

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        plt.title('Pareto Fronts')

        plt.show()

    def mutation(self, hyperparameter):
        """Choose hyperparameters to random the value"""
        result = hyperparameter.copy()
        if np.random.rand() < MUTATION_RATE:
            is_mutate = np.random.choice([0, 1, 2])
            if is_mutate == 0:
                result['learning_rate'] = generate_random_number(LEARNING_RATE)
            elif is_mutate == 1:
                result['n_estimators'] = generate_random_number(N_ESTIMATORS, is_int=True)
            else:
                result['learning_rate'] = generate_random_number(LEARNING_RATE)
                result['n_estimators'] = generate_random_number(N_ESTIMATORS, is_int=True)

        return MVPPredictionModel(self.nba_stats, self.objectives, result)

    def optimize(self):
        pass
