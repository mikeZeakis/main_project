import numpy as np 
import cvxpy as cp

def dea_input_oriented_extended(X, Y, dmu_index, epsilon=1e-4):
    """
    Solve the input-oriented DEA problem with slacks for a given DMU.

    Parameters:
    - X: numpy array of shape (n_dmus, n_inputs)
    - Y: numpy array of shape (n_dmus, n_outputs)
    - dmu_index: index of the DMU to evaluate (0-based)
    - epsilon: small constant to penalize slacks

    Returns:
    - theta value (efficiency score)
    - input slacks s_minus
    - output slacks s_plus
    - lambdas used to form reference DMU
    """
    n_dmus, n_inputs = X.shape
    _, n_outputs = Y.shape

    # Decision variables
    theta = cp.Variable()
    lambdas = cp.Variable(n_dmus)
    s_minus = cp.Variable(n_inputs)
    s_plus = cp.Variable(n_outputs)

    # Objective function
    objective = cp.Minimize(theta - epsilon * (cp.sum(s_minus) + cp.sum(s_plus)))

    # Constraints
    constraints = []

    # Input constraints: theta * x_o - X^T @ lambda - s_minus = 0
    x_o = X[dmu_index]
    y_o = Y[dmu_index]
    for i in range(n_inputs):
        constraints.append(theta * x_o[i] - X[:, i] @ lambdas - s_minus[i] == 0)

    # Output constraints: Y^T @ lambda - s_plus = y_o
    for j in range(n_outputs):
        constraints.append(Y[:, j] @ lambdas - s_plus[j] == y_o[j])

    # Non-negativity
    constraints += [lambdas >= 0, s_minus >= 0, s_plus >= 0]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return theta.value, s_minus.value, s_plus.value, lambdas.value


def visualize_efficiencies():
    
    return 0