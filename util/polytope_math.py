import numpy as np
from scipy.optimize import minimize


def extract_plane_eqn_from_coeffs(coeffs: list):
    r"""
    Extract plane equations from halfspace coefficients
        :param coeffs: list of dictionaries containing halfspace coefficients
        a, b, c, d in the equation ax + by + cz + d = 0
    """
    H = np.zeros((len(coeffs), 3))
    d_vec = np.zeros((len(coeffs),))
    i = 0
    for h in coeffs:
        H[i] = np.array([h['a'], h['b'], h['c']])
        d_vec[i] = h['d']
        i += 1
    return H, d_vec

def get_closest_distance_to_polytope_surface(point: np.ndarray,
                                             A: np.ndarray,
                                             b: np.ndarray) -> float:
    """
    Computes the shortest distance from a point to the surface of a polytope.

    Args:
        point (np.ndarray): The point to measure the distance from.
        A (np.ndarray): The matrix defining the linear inequalities of the polytope (Ax <= b).
        b (np.ndarray): The vector defining the linear inequalities of the polytope (Ax <= b).

    Returns:
        float: The shortest distance to the surface of the polytope.
    """
    num_facets = A.shape[0]
    distances = []

    # Iterate over each facet of the polytope
    for i in range(num_facets):
        # Define the objective function (squared distance from the point)
        def objective(x):
            return np.sum((x - point) ** 2)

        # Define the constraints for the current facet
        # The point must lie on the current facet (equality constraint)
        facet_eq_constraint = {'type': 'eq', 'fun': lambda x: A[i, :] @ x - b[i]}

        # The point must also be within the other half-spaces (inequality constraints)
        other_ineq_constraints = []
        for j in range(num_facets):
            if i != j:
                other_ineq_constraints.append({'type': 'ineq', 'fun': lambda x, k=j: b[k] - A[k, :] @ x})

        # Initial guess for the optimization
        x0 = point

        # Solve the constrained optimization problem
        result = minimize(
            objective,
            x0,
            constraints=[facet_eq_constraint] + other_ineq_constraints,
            method='SLSQP'
        )

        # If the optimization was successful, calculate the distance
        if result.success:
            distance_to_facet = np.sqrt(result.fun)
            distances.append(distance_to_facet)
        else:
            # Handle cases where the optimization fails for a facet (e.g., empty facet)
            pass

    # The shortest distance to the surface is the minimum of the distances to all facets
    if distances:
        return np.min(distances)
    else:
        return np.nan # No valid distance found