import numpy as np
import sourcedefender
from HomeworkFramework import Function
import math
import pandas as pd
import os
import sys

class CMAES_optimizer(Function):
    def __init__(self, target_func):
        super().__init__(target_func)
        # number of objective variables/problem dimension
        self.dim = self.f.dimension(target_func) 
        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        print(self.lower, self.upper, self.dim)
        print("bang------------------bang")
        
        self.target_func = target_func
        self.eval_times = 0 # equivalent to g , g is number of generations
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)
        self.target_func = target_func
        self.setting()
    
    def init_setting(self):
        #------my parameter------#
        
        # population size, offspring number
        self.Lambda = 5 + int(3 * np.log(self.dim))
        
        # evolution paths for c_matrix and sigma
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
        # objective variables initial point
        self.xmean = np.copy(np.random.random(self.dim))
        
        # coordinate wise standard deviation (step-size)
        self.sigma = 0.25
        
        # c_sys : coordinate system
        # scaling_c_matrix : is a diagonal matrix, defines the scaling covariance matrix.
        self.c_sys = np.eye(self.dim) 
        self.c_matrix = np.eye(self.dim)
        self.scaling_c_matrix = np.ones(self.dim)
        
    def setting(self):
        self.init_setting()
        
        #this is for mutation
        self.mutation = int(self.Lambda / 2)
        
        self.weights = np.array([np.log(self.mutation + 0.5) - np.log(i + 1) for i in range(self.mutation)])
        
        # normalize
        self.weights = np.array([i / sum(self.weights) for i in self.weights])
        # variance-effective size of mutation
        tmp = np.sum(np.power(i, 2) for i in self.weights)
        self.mu_param = 1 / tmp
        
        # time constant for cumulation for c_matrix
        tmp = (4 + self.mu_param / self.dim)
        tmp1 = (self.dim + 3 + 2 * self.mu_param / self.dim)
        self.c_cumulate = tmp / tmp1
        
        # t-const for cumulation for sigma control
        tmp = (self.mu_param + 6)
        tmp1 = (self.dim + self.mu_param + 4)
        self.c_sig = tmp / tmp1
        
        # learning rate for rank-one update of c_matrix
        tmp = ((self.dim + 1.3) ** 2 + self.mu_param)
        self.c_one_rate = 2 / tmp
        # for rank-mutation update
        self.c_mutation = min([1 - self.c_one_rate, 2 * ((self.mu_param-2) + (1/self.mu_param)) / ((self.dim + 2) ** 2 + self.mu_param)])
        
          
    def fit(self, FES):
        fitness = np.zeros((self.Lambda, self.dim))
        arz = np.zeros((self.Lambda, self.dim))
        arx = np.zeros((self.Lambda, self.dim))
        fitvals = np.zeros(self.Lambda)
        for i in range(self.Lambda):
            #if i < FES:
            
            arz[i] = np.random.normal(0, 1, self.dim)
            arx[i] = np.dot(self.c_sys * self.scaling_c_matrix, arz[i])
            
            fitness[i] = self.xmean + self.sigma * arx[i]
            fitness[i] = np.clip(fitness[i], self.lower, self.upper)
            #print(fitness[i])
            
            value = self.f.evaluate(func_num, fitness[i])
            self.eval_times += 1
            
            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break
                
            fitvals[i] = value
            if self.optimal_value > fitvals[i]:
                self.optimal_value = fitvals[i]
                self.optimal_solution = fitness[i]
                
        return fitness, arz, arx, fitvals, value
    
    def update_pc_ps(self, arz, arx, argx):
        # update evolution path
        tmp = np.sum(self.weights[i] * arz[argx[i]] for i in range(self.mutation))
        ps_bias = tmp*np.sqrt(self.c_sig * (2 - self.c_sig) * self.mu_param)
        self.ps = self.ps - self.c_sig * self.ps + ps_bias
        
        tmp = np.sum(self.weights[i] * arx[argx[i]] for i in range(self.mutation))
        pc_bias = tmp*np.sqrt(self.c_cumulate * (2 - self.c_cumulate) * self.mu_param)
        self.pc = self.pc - self.c_cumulate * self.pc + pc_bias
        
    def update_c_matrix(self, arx, argx):
        # update covariance matrix c_matrix
        tmp = np.zeros((self.dim, self.dim))
        for i in range(self.mutation):
            tmp += self.c_mutation * self.weights[i] * np.dot(arx[argx[i]].reshape(self.dim, 1), arx[argx[i]].reshape(1, self.dim))
        self.c_matrix = (1 - self.c_one_rate - self.c_mutation) * self.c_matrix
        self.c_matrix += self.c_one_rate * np.dot(self.pc.reshape(self.dim, 1), self.pc.reshape(1, self.dim))
        self.c_matrix += tmp
        
    def update_related_param(self, arx, argx):
        self.xmean = self.xmean + self.sigma * np.sum(self.weights[i] * arx[argx[i]] for i in range(self.mutation))
        # update step-size
        self.sigma *= np.exp((self.c_sig / 2) * (np.sum(np.power(x, 2) for x in self.ps) / self.dim - 1))
        
    def CMAES_method(self, FES):
        # Sample
        self.scaling_c_matrix, self.c_sys = np.linalg.eigh(self.c_matrix)
        self.scaling_c_matrix = self.scaling_c_matrix** 0.5
        
        fitness, arz, arx, fitvals, value = self.fit(FES)

        if value == "ReachFunctionLimit":
            print("ReachFunctionLimit")
            return
        
        # sort and update mean
        argx = np.argsort(fitvals)

        self.update_pc_ps(arz, arx, argx)       
        self.update_c_matrix(arx, argx)
        self.update_related_param(arx, argx) 

    def run(self, FES):
        while self.eval_times < FES:
            #print('=====================FE=====================')
            #print(self.eval_times)
            self.CMAES_method(FES)
            
            #print("optimal: {}\n".format(self.get_optimal()[1]))

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000 
        else:
            fes = 2500
        
        print("--------start----------")
        # you should implement your optimizer
        op =  CMAES_optimizer(func_num)
        op.run(fes)
        '''
        [ 0.4999504  -0.50109196  0.503416   -0.52337186  0.51543136 -0.43618353] 1.5196545412161856e-08
        
        [1. 1.] 1.3497838043956716e-31
        
        [1.29569686e-15 1.00000000e+00 2.00000000e+00 3.00000000e+00
 4.00000000e+00] 3.952393967665557e-14
        
        [-9.99984287e+00  1.04438615e+02 -2.09136469e-04  7.70000877e+01
  5.89928796e+01 -3.19999873e+01 -4.29999656e+01  5.09995155e+01
  9.00001193e+01 -1.19999244e+01] 0.01723615883904539
        '''
        
        best_input, best_value = op.get_optimal()
        print(best_input, best_value)
        print("--------end----------")
        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1 