import cvxpy

import numpy as np
import pandas as pd

# from .utils.maximum_likelihood_estimator import


dataset = pd.ExcelFile('../dataset/data.xlsx')
requested_data = pd.read_excel(dataset, 'requested')
picked_data = pd.read_excel(dataset, 'picked')
used_data = pd.read_excel(dataset, 'used')
returned_data = pd.read_excel(dataset, 'returned')

number_of_items, number_of_operations = requested_data.shape
number_of_operations -= 3

requested_data = requested_data.fillna(0)
picked_data = picked_data.fillna(0)
used_data = used_data.fillna(0)
returned_data = returned_data.fillna(0)

# Constructing the request matrix
A = np.transpose(np.array(requested_data)[:, 3:].astype(np.int))

# Constructing the usage matrix
B = np.transpose(np.array(used_data)[:, 3:].astype(np.int))

# Constructing the return matrix
D = A - B
R = 0.5 * (np.abs(D) + D)

# Constructing the dynamic request matrix
E = 0.5 * (np.abs(D) - D)

X = cvxpy.Variable((number_of_items, 1), integer=True)
Y = cvxpy.Variable((number_of_items, number_of_operations), integer=True)
Z = cvxpy.Variable((number_of_items, number_of_operations), integer=True)

# Determining the request cost, dynamic request cost, return cost
# C = np.random.rand(number_of_items, 1)
# C_d = np.random.rand(number_of_items, 1)

F = np.array([1]*number_of_items)
C = np.array([1]*number_of_items)
Q = np.array([1]*number_of_items)

f1 = cvxpy.matmul(np.transpose(F), X)

# Probability of each scenario
Ps = 1.0 / number_of_operations

f2 = 0
for i in range(number_of_operations):
    f2 += Ps * cvxpy.matmul(np.transpose(C), Y[:, i])

f3 = 0
for i in range(number_of_operations):
    f2 += Ps * cvxpy.matmul(np.transpose(Q), Z[:, i])


upper_bound = np.zeros((number_of_items, 1))
lower_bound = np.zeros((number_of_items, 1))

e_upper_bound = np.zeros((number_of_items, 1))

for i in range(B.shape[1]):
    upper_bound[i, 0] = max(B[:, i])
    lower_bound[i, 0] = min(B[:, i])
    e_upper_bound[i, 0] = max(E[:, i])


objective = cvxpy.Minimize(f1 + f2 + f3)
constraints = [Y <= np.transpose(A), Z >= 0, Y >= 0, X >= lower_bound, X <= upper_bound]

# Additional Constraints
# X >= e_upper_bound makes sure that we wont have shortage


problem = cvxpy.Problem(objective, constraints)
problem.solve(solver=cvxpy.GLPK_MI, verbose=True)
print('[*] Minimum of the objective function')
print(problem.value)
print('[*] Optimum X vector')
print(X.value)

# for i in range(number_of_operations):
#     print('\t', sum(X.value - B[i, :].reshape(B.shape[1], 1)))
ans = np.concatenate((X.value, B[-1, :].reshape(-1, 1)), axis=1)
print('[*] Comparison between the last use vector and the optimized vector (left: optimum, right: last operation')
print(ans)
# print(sum(X.value - B[-1, :].reshape(B.shape[1], 1)))

print('[*] The average used items')
print(np.round(np.mean(B, axis=0)))
