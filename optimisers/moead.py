import numpy as np
from optimisers.moea import MOEA
from pymoo.util.ref_dirs import get_reference_directions
from utils import euclidean_distance_from_point_to_vector

class MOEAD(MOEA):
    def __init__(self, nba_stats, pop_size, n_evaluations, n_partitions, n_neighbours, mvp_share_rmse=False, execution_time=False, mvp_ranking_rmse=False, population=[]) -> None:
        super().__init__(nba_stats, pop_size, n_evaluations, mvp_share_rmse, execution_time, mvp_ranking_rmse, population)
        self.weights = get_reference_directions(
            "uniform",
            self.n_objectives,
            n_partitions=n_partitions
        )
        self.n_weights = len(self.weights)
        self.T_ = n_neighbours
        self.solutions = np.array([
            solution.get_fitness() for solution in self.population
        ])

        self.z = self.init_z()
        self.b = self.generate_closest_weight_vectors()
        self.nearest_weight = [-1] * self.size_p
        self.w_solutions = self.init_objective_nearest_weight_vector()

    def init_objective_nearest_weight_vector(self):
        w_solutions = [[]] * self.n_weights
        for i, solution in enumerate(self.solutions):
            min_distance = np.inf
            nearest_weight_index = -1
            for j, w in enumerate(self.weights):
                distance = euclidean_distance_from_point_to_vector(
                    point=solution, 
                    start=[0] * self.n_objectives,
                    end=w
                )
                if min_distance > distance:
                    min_distance = distance
                    nearest_weight_index = j
            w_solutions[nearest_weight_index].append(i)
            self.nearest_weight[i] = nearest_weight_index
        return w_solutions

    def init_z(self):
        z = []

        for i in range(self.n_objectives):
            z.append(self.solutions[:, i].min())

        return np.array(z)

    def generate_closest_weight_vectors(self):
        b = []
        for i in range(self.n_weights):
            b_i = np.zeros(self.T_, int)
            b_dist = np.full(self.T_, np.inf)
            for j in range(self.n_weights):
                dist_wi_wj = np.linalg.norm(np.array(self.weights[i]) - np.array(self.weights[j]))
                if dist_wi_wj < np.max(b_dist):
                    index_to_replace = np.argmax(b_dist)  # replace the worst distance
                    b_dist[index_to_replace] = dist_wi_wj
                    b_i[index_to_replace] = j

            b.append(b_i.tolist())

        return b

    def tchebycheff(self, solution, sub_problem):
        max_distance = 0
        for i in range(self.n_objectives):
            current_score = abs(solution[i] - self.z[i]) * self.weights[sub_problem][i]
            max_distance = max(max_distance, current_score)
        return max_distance

    def update_z(self, new_solution):
        for i in range(self.n_objectives):
            if new_solution[i] < self.z[i]:
                self.z[i] = new_solution[i]

    def selection_operator(self, sub_problem):
        g_min = np.inf
        index = -1
        for i in self.w_solutions[sub_problem]:
            g = self.tchebycheff(self.solutions[i], sub_problem)
            if g_min > g:
                g_min = g
                index = i
        return self.population[index]

    def replacement(self, solution):
        replace_index = -1
        for front in self.fronts[::-1]:
            if replace_index != -1:
                break
            for j in front:
                if solution >= self.population[j]:
                    replace_index = j
                    break

        if replace_index == -1:
            return False

        min_distance = np.inf
        nearest_weight_index = -1
        for j, w in enumerate(self.weights):
            distance = euclidean_distance_from_point_to_vector(
                point=solution.get_fitness(), 
                start=[0] * self.n_objectives,
                end=w
            )
            if min_distance > distance:
                min_distance = distance
                nearest_weight_index = j

        w_index = self.w_solutions[self.nearest_weight[replace_index]].index(replace_index)
        self.w_solutions[self.nearest_weight[replace_index]].pop(w_index)

        self.population[replace_index] = solution
        self.solutions[replace_index] = solution.get_fitness()
        self.update_z(solution.get_fitness())

        self.w_solutions[nearest_weight_index].append(replace_index)
        self.nearest_weight[replace_index] = nearest_weight_index

        return True

    def optimize(self):
        is_break = False

        while True:
            self.non_dominated_sorting()
            self.calc_performance_metric()
            for b in self.b:
                k = np.random.choice(b)
                l = np.random.choice(b)
                parent_a = self.selection_operator(k)
                parent_b = self.selection_operator(l)
                childrens = parent_a.crossover(parent_b)
                new_solution_c = self.mutation(childrens[0])
                self.replacement(new_solution_c)
                self.eval_count += 1
                print('\tEvaluation: ', self.eval_count)
                if self.eval_count == self.n_evaluations:
                    is_break=True
                    break
                new_solution_d = self.mutation(childrens[1])
                self.replacement(new_solution_d)
                self.eval_count += 1
                print('\tEvaluation: ', self.eval_count)
                if self.eval_count == self.n_evaluations:
                    is_break = True
                    break
            if is_break:
                break
        self.non_dominated_sorting()
        self.calc_performance_metric()
