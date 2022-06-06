import numpy as np

# you must use python 3.6, 3.7, 3.8(3.8 not for macOS) for sourcedefender
import sourcedefender
from HomeworkFramework import Function

class CoDE_optimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func)  # must have this init to work normally

        self.dim = self.f.dimension(target_func)
        self.lower = self.f.lower(target_func) * np.ones(self.dim)
        self.upper = self.f.upper(target_func) * np.ones(self.dim)
        self.target_func = target_func
        self.eval_times = 0
        self.generations = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        self.set_parameters()

        self.parents = np.full((self.population_size,self.dim), None)
        self.parents_objective_values = np.full(self.population_size, None)

        self.offsprings = np.full((self.population_size,self.dim), None)
        self.offsprings_objective_values = np.full(self.population_size, None)

    def set_parameters(self):
        self.population_size = 10
        self.strategy_candidate_pool = ["rand/1/bin", "rand/2/bin", "current-to-rand/1"]
        self.parameter_candidate_pool = [(1.0, 0.1),(1.0, 0.9),(0.8, 0.2)]

    def initial_population(self):
        for i in range(len(self.parents)):
            self.parents[i] = np.random.uniform(self.lower, self.upper, self.dim)
            #self.eval_times+=1
        #self.parents_objective_values = self.target_func(self.parents)
        for i in range(len(self.parents)):
            #self.parents_objective_values[i] = self.target_func(self.parents[i])
            self.parents_objective_values[i] = self.f.evaluate(self.target_func, self.parents[i])
            self.eval_times += 1
        min_index = np.argmin(self.parents_objective_values)
        if self.parents_objective_values[min_index]<self.optimal_value:
            self.optimal_solution[:] = self.parents[min_index]
            self.optimal_value = self.parents_objective_values[min_index]

    def produce(self):
        trial_vectors = np.full((self.population_size*len(self.strategy_candidate_pool),self.dim), None)
        trial_vectors_objective_values = np.full(self.population_size*len(self.strategy_candidate_pool), float("inf"))
        for i in range(self.population_size):
            j_rand = np.random.randint(self.dim)
            r1, r2, r3, r4, r5 = np.random.choice(self.population_size, size=5, replace=False)
            index = 0
            #rand/1/bin
            para = np.random.randint(len(self.parameter_candidate_pool))
            for j in range(self.dim):
                if np.random.rand()<self.parameter_candidate_pool[para][1] or j==j_rand:
                    trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = self.parents[r1][j] + self.parameter_candidate_pool[para][0]*(self.parents[r2][j]-self.parents[r3][j])
                    if trial_vectors[i*len(self.strategy_candidate_pool)+index][j]< self.lower[j]:
                        trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = min(self.upper[j], 2*self.lower[j]-trial_vectors[i*len(self.strategy_candidate_pool)+index][j])
                    elif trial_vectors[i*len(self.strategy_candidate_pool)+index][j]>self.upper[j]:
                        trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = max(self.lower[j], 2*self.upper[j]-trial_vectors[i*len(self.strategy_candidate_pool)+index][j])
                else:
                    trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = self.parents[i][j]
            index+=1

            #rand/2/bin
            para = np.random.randint(len(self.parameter_candidate_pool))
            for j in range(self.dim):
                if np.random.rand() < self.parameter_candidate_pool[para][1] or j == j_rand:
                    trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = self.parents[r1][j] + self.parameter_candidate_pool[para][0] * (
                    self.parents[r2][j] - self.parents[r3][j]) + self.parameter_candidate_pool[para][0]*(
                    self.parents[r4][j] - self.parents[r5][j])
                    if trial_vectors[i*len(self.strategy_candidate_pool)+index][j]< self.lower[j]:
                        trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = min(self.upper[j], 2*self.lower[j]-trial_vectors[i*len(self.strategy_candidate_pool)+index][j])
                    elif trial_vectors[i*len(self.strategy_candidate_pool)+index][j]>self.upper[j]:
                        trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = max(self.lower[j], 2*self.upper[j]-trial_vectors[i*len(self.strategy_candidate_pool)+index][j])
                else:
                    trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = self.parents[i][j]
            index += 1

            #current-to-rand/1
            para = np.random.randint(len(self.parameter_candidate_pool))
            coefficient = np.random.rand()
            trial_vectors[i*len(self.strategy_candidate_pool)+index] = self.parents[i] + coefficient*(self.parents[r1]-self.parents[i]) + coefficient*self.parameter_candidate_pool[para][0]*(self.parents[r2]-self.parents[r3])
            for j in range(self.dim):
                if trial_vectors[i*len(self.strategy_candidate_pool)+index][j] < self.lower[j]:
                    trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = min(self.upper[j], 2 * self.lower[j] - trial_vectors[i*len(self.strategy_candidate_pool)+index][j])
                elif trial_vectors[i*len(self.strategy_candidate_pool)+index][j] > self.upper[j]:
                    trial_vectors[i*len(self.strategy_candidate_pool)+index][j] = max(self.lower[j], 2 * self.upper[j] - trial_vectors[i*len(self.strategy_candidate_pool)+index][j])

        #trial_vectors_objective_values = self.target_func(trial_vectors)
        #self.eval_times+=(self.population_size*len(self.strategy_candidate_pool))
        for i in range(len(trial_vectors)):
            #trial_vectors_objective_values[i] = self.target_func(trial_vectors[i])
            objective_value = self.f.evaluate(self.target_func, trial_vectors[i])
            self.eval_times += 1
            if objective_value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break
            trial_vectors_objective_values[i] = objective_value

        for i in range(self.population_size):
            index = range(i*len(self.strategy_candidate_pool),i*len(self.strategy_candidate_pool)+3)
            min_index = index[np.argmin(trial_vectors_objective_values[i*len(self.strategy_candidate_pool):i*len(self.strategy_candidate_pool)+3])]
            self.offsprings[i][:] = trial_vectors[min_index]
            self.offsprings_objective_values[i] = trial_vectors_objective_values[min_index]

    def select(self):
        for i in range(self.population_size):
            if self.offsprings_objective_values[i] < self.parents_objective_values[i]:
                self.parents[i][:] = self.offsprings[i]
                self.parents_objective_values[i] = self.offsprings_objective_values[i]
                if self.parents_objective_values[i] < self.optimal_value:
                    self.optimal_solution[:] = self.parents[i]
                    self.optimal_value = self.parents_objective_values[i]

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES): # main part for your implementation
        self.initial_population()
        while self.eval_times<FES:
            print('======= CoDE Generation {', int(self.generations), '} =====================')
            print('=====================FE============================')
            print(self.eval_times)

            self.produce()
            self.select()
            print("optimal: {}\n".format(self.get_optimal()[1]))
            self.generations+=1

if __name__ == '__main__':
    func_num = 1
    fes = 0
    # function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500

        # you should implement your optimizer
        op = CoDE_optimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(best_input, best_value)

        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__, func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
