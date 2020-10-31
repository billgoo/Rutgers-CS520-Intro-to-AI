from utils import *
import numpy as np


class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a beta value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    """
        states NEW as 0, DEAD as 9, and USEDi as i
        actions USED as 0, REPLACE as 1
    """

    t_used, t_replace, reward_used, reward_replace = [], [], [100], [0]

    def __init__(self, beta=.9):
        # def __init__(self, init, actlist, terminals, beta=.9):
        '''
        update(self, init=init, actlist=actlist, terminals=terminals,
               beta=beta, states=set(), reward={})
               '''
        self.beta = beta
        self.build_model()

    def build_model(self):
        self.t_used.append([(1, 1)])
        for i in range(1, 9):
            p = 0.1 * i
            self.t_used.append([(p, i + 1), (1 - p, i)])
        self.t_used.append([(float('-inf'), 0)])
        self.t_replace.append([(float('-inf'), 0)])
        for i in range(1, 10):
            # replace will all go to New
            self.t_replace.append([(1, 0)])
        # replace will have the unchanged cost so don't need to specific
        # reward for use
        for i in range(1, 9):
            self.reward_used.append(100 - 10 * i)
        self.reward_used.append(0)
        for i in range(1, 10):
            self.reward_replace.append(-250)

    def R(self):
        """Return a numeric reward for this state."""
        return [self.reward_used, self.reward_replace]

    def T(self):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        result = [self.t_used, self.t_replace]
        return result

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state == 0:
            return [0]
        elif state == 9:
            return [1]
        else:
            return [0, 1]

    def __del__(self):
        print("__del__")



def value_iteration(mdp, epsilon=0.001):
    """Solving an MDP by value iteration."""
    U1 = dict([(i, 0) for i in range(10)])
    R, T, beta = mdp.R(), mdp.T(), mdp.beta
    while True:
        U = U1.copy()
        delta = 0
        for s in range(10):
            '''
            print(s)
            print(mdp.actions(s))
            print([(R[a][s] + beta * sum([p * U[s1] for (p, s1) in T[a][s]]))
                         for a in mdp.actions(s)])
                         '''
            U1[s] = max([(R[a][s] + beta * sum([p * U[s1] for (p, s1) in T[a][s]]))
                         for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - beta) / beta:
            return U


def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in range(10):
        # for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a: expected_utility(a, s, U, mdp))

    return pi


def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s, according to the MDP and U."""
    # print(sum([p * U[s1] for (p, s1) in mdp.T()[a][s]]))
    R, T, beta = mdp.R(), mdp.T(), mdp.beta
    return R[a][s] + beta * sum([p * U[s1] for (p, s1) in T[a][s]])


'''
def policy_iteration(mdp):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s, 0) for s in mdp.states])
    pi = dict([(s, random.choice(mdp.actions(s))) for s in mdp.states])
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), lambda a: expected_utility(a, s, U, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi


def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, beta = mdp.R, mdp.T, mdp.beta
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + beta * sum([p * U[s] for (p, s1) in T(s, pi[s])])
    return U
'''


def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    argmax(['one', 'to', 'three'], len)
    'three'
    return 0: used, 1: replace, 2: both
    """
    if len(seq) == 1:
        return seq[0]
    best = -1
    best_score = float("-inf")
    for x in seq:
        x_score = fn(x)
        if x_score > best_score:
            best, best_score = x, x_score
        elif x_score == best_score:
            best, best_score = 2, x_score
    return best


if __name__ == '__main__':
    mdp = MDP()
    U = value_iteration(mdp)
    print("U:", U)
    print("policy:", best_policy(mdp, U))
    print(mdp.T())
    print(mdp.R())
    del mdp
