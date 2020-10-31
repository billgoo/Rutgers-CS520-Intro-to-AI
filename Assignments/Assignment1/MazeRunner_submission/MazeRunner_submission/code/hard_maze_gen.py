import numpy
import copy
import random
import pandas
from MapGen import maze_generate
from DFS import dfs
from A_star import *
from BFS import *


class GA(object):
    def __init__(self, running_tiems, dim, p, goal, population_size, pc, pm, pn):
        self.running_times = running_tiems
        self.dim = dim
        self.p = p
        self.goal = goal
        self.population_size = population_size
        self.pc = pc
        self.pm = pm
        self.pn = pn

    def maze_origin(self):
        population = []
        for i in range(self.population_size):
            population.append(maze_generate(self.dim, self.p))
        return population

    def evaluation(self, population):
        cmin = self.dim * self.dim
        fitness = [0 for i in range(len(population))]
        fitness_cum = [0 for i in range(len(population))]
        for i in range(len(population)):
            solution_bfs = bfs(population[i])
            if len(solution_bfs['PATH']) == 0:  # if there is no path, ignore this maze
                fitness[i] = 0
                continue
            solution_dfs = dfs(population[i])
            solution_man = a_star_manhattan(population[i])
            solution_euc = a_star_euclidean(population[i])
            if self.goal == 'PATH':
                fitness[i] = len(solution_bfs[self.goal]) + len(solution_dfs[self.goal]) + \
                             len(solution_man[self.goal]) + len(solution_euc[self.goal])
            elif self.goal == 'NODE':
                fitness[i] = len(solution_bfs[self.goal]) + solution_dfs[self.goal] + \
                             int(solution_man[self.goal]) + int(solution_euc[self.goal])
            else:
                fitness[i] = solution_bfs[self.goal] + solution_dfs[self.goal] + \
                             solution_man[self.goal] + solution_euc[self.goal]
            if fitness[i] < cmin:
                cmin = fitness[i]

        for i in range(len(fitness)):
            fitness_cum[i] = fitness[i] - cmin + 1
            if fitness_cum[i] < 0:
                fitness_cum[i] = 0
        probability = fitness_cum / numpy.sum(fitness_cum)
        cum_probability = numpy.cumsum(probability)
        return fitness, cum_probability

    def selection(self, population, cum_probability):
        survival = []
        for i in range(self.population_size):
            survival.append(random.random())
        survival.sort()
        new_population = []
        org = 0
        new = 0
        while new < self.population_size:
            if survival[new] < cum_probability[org]:
                new_population.append(population[org])
                new += 1
            else:
                org += 1
        population[:] = new_population

    def crossover(self, population):
        size = len(population)
        for i in range(size - 1):
            if random.random() < self.pc:
                row_position = random.randint(0, self.dim - 1)
                col_position = random.randint(0, self.dim - 1)
                temp1 = numpy.row_stack(
                    (population[i][0:row_position], population[i + 1][row_position:self.dim]))
                temp2 = numpy.row_stack(
                    (population[i + 1][0:row_position], population[i][row_position:self.dim]))
                temp3 = numpy.column_stack((temp1[:, 0:col_position], temp2[:, col_position: self.dim]))
                temp4 = numpy.column_stack((temp2[:, 0:col_position], temp1[:, col_position: self.dim]))
                population.append(temp3)
                population.append(temp4)

    def mutation(self, population):
        size = len(population)
        for i in range(size):
            if random.random() < self.pm:
                mut = copy.deepcopy(population[i])
                for j in range(int(self.pn * self.dim * self.dim)):
                    position = [random.randint(0, self.dim - 1), random.randint(0, self.dim - 1)]
                    if mut[position[0]][position[1]] == 1:
                        mut[position[0]][position[1]] = 0
                    else:
                        mut[position[0]][position[1]] = 1
                mut[0][0] = 0
                mut[len(mut) - 1][len(mut[0]) - 1] = 0
                population.append(mut)

    def best(self, population, fitness):
        best_fitness = 0
        best_individual = []
        for i in range(len(population)):
            if best_fitness < fitness[i]:
                best_fitness = fitness[i]
                best_individual = population[i]
        return best_individual, best_fitness

    def performance(self, maze, all_dfs, all_bfs, all_euc, all_man):
        solution_bfs = bfs(maze)
        solution_dfs = dfs(maze)
        solution_man = a_star_manhattan(maze)
        solution_euc = a_star_euclidean(maze)

        all_dfs[0].append(len(solution_dfs['PATH']))
        all_dfs[1].append(solution_dfs['NODE'])
        all_dfs[2].append(solution_dfs['FRINGE'])

        all_bfs[0].append(len(solution_bfs['PATH']))
        all_bfs[1].append(len(solution_bfs['NODE']))
        all_bfs[2].append(solution_bfs['FRINGE'])

        all_euc[0].append(len(solution_euc['PATH']))
        all_euc[1].append(int(solution_euc['NODE']))
        all_euc[2].append(solution_euc['FRINGE'])

        all_man[0].append(len(solution_man['PATH']))
        all_man[1].append(int(solution_man['NODE']))
        all_man[2].append(solution_man['FRINGE'])

    def main(self):
        all_best = []
        all_dfs = [[], [], []]
        all_bfs = [[], [], []]
        all_euc = [[], [], []]
        all_man = [[], [], []]
        population = self.maze_origin()
        for i in range(self.running_times):
            fitness, cum_probability = self.evaluation(population)
            best_individual, best_fitness = self.best(population, fitness)

            # print processing information
            self.performance(best_individual, all_dfs, all_bfs, all_euc, all_man)
            all_best.append(best_fitness)
            if i % 100 == 0:
                numpy.save(self.goal + '\\' + str(i) + '_hard_maze_' + goal + '.npy', best_individual)
            print('No.', i, ', Goal:', self.goal, best_fitness)
            # end print

            self.selection(population, cum_probability)
            self.crossover(population)
            self.mutation(population)

        df = pandas.DataFrame({self.goal: all_best,
                               'dfs_path': all_dfs[0], 'dfs_node': all_dfs[1], 'dfs_fringe': all_dfs[2],
                               'bfs_path': all_bfs[0], 'bfs_node': all_bfs[1], 'bfs_fringe': all_bfs[2],
                               'euc_path': all_euc[0], 'euc_node': all_euc[1], 'euc_fringe': all_euc[2],
                               'man_path': all_man[0], 'man_node': all_man[1], 'man_fringe': all_man[2]})
        df.to_csv(self.goal + '\\' + 'process_info.csv', index=True)

        print('Finished!')
        return best_individual


if __name__ == '__main__':
    running_times = 1000
    dim = 25
    p = 0.2
    goal = 'NODE'  # PATH, NODE, FRINGE
    population_size = 100
    pc = 0.8
    pm = 0.05
    pn = 0.01
    ga = GA(running_times, dim, p, goal, population_size, pc, pm, pn)
    best_individual = ga.main()
    numpy.save('data/'+goal + '/' + 'hardest_maze_' + goal + '.npy', best_individual)
