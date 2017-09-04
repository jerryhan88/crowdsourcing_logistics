from gurobipy import *

m = Model('')
m.read('/Users/JerryHan88/PycharmProjects/crowdsourcing_logistics/subProblem0.lp')
m.optimize()