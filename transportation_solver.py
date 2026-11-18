"""
Operations Research - Transportation Cost Minimization Solvers
Implements:
1. Northwest Corner Rule (NWC)
2. Least Cost Method (LCM / Matrix Minimum)
3. Vogel's Approximation Method (VAM)
4. Linear Programming Optimal Benchmark (SciPy linprog)
"""

import numpy as np
from scipy.optimize import linprog


def calculate_cost(allocation, costs):
    """Calculates total transportation cost sum(alloc_ij * cost_ij)."""
    return int(np.sum(allocation * costs))


def format_breakdown(allocation, costs, row_names=None, col_names=None):
    """Generates detailed arithmetic breakdown and active shipment list."""
    m, n = costs.shape
    if row_names is None:
        row_names = [f"Factory {i+1}" for i in range(m)]
    if col_names is None:
        col_names = [f"Warehouse {chr(65+j)}" for j in range(n)]

    shipments = []
    expr_parts = []
    total_cost = 0

    for i in range(m):
        for j in range(n):
            qty = int(allocation[i, j])
            if qty > 0:
                unit_cost = int(costs[i, j])
                line_cost = qty * unit_cost
                total_cost += line_cost
                shipments.append({
                    "Source": row_names[i],
                    "Destination": col_names[j],
                    "Quantity": qty,
                    "Unit Cost": unit_cost,
                    "Total Cost": line_cost,
                })
                expr_parts.append(f"({qty} × {unit_cost})")

    calc_string = " + ".join(expr_parts) + f" = {total_cost} cost units" if expr_parts else "0 cost units"
    return shipments, calc_string, total_cost


def solve_northwest_corner(supply, demand, costs):
    """Solves initial BFS using the Northwest Corner Rule."""
    s = np.array(supply, dtype=float).copy()
    d = np.array(demand, dtype=float).copy()
    m, n = costs.shape
    allocation = np.zeros((m, n), dtype=int)

    i, j = 0, 0
    while i < m and j < n:
        qty = min(s[i], d[j])
        allocation[i, j] = int(qty)
        s[i] -= qty
        d[j] -= qty

        if s[i] == 0 and d[j] == 0:
            if i + 1 < m:
                i += 1
            else:
                j += 1
        elif s[i] == 0:
            i += 1
        else:
            j += 1

    return allocation, calculate_cost(allocation, costs)


def solve_least_cost(supply, demand, costs):
    """Solves initial BFS using the Least Cost Method (Matrix Minimum)."""
    s = np.array(supply, dtype=float).copy()
    d = np.array(demand, dtype=float).copy()
    c = np.array(costs, dtype=float).copy()
    m, n = c.shape
    allocation = np.zeros((m, n), dtype=int)

    row_active = [True] * m
    col_active = [True] * n

    while any(row_active) and any(col_active):
        # Mask out exhausted rows and columns with infinity
        masked = np.full((m, n), np.inf)
        for i in range(m):
            if row_active[i]:
                for j in range(n):
                    if col_active[j]:
                        masked[i, j] = c[i, j]

        if np.isinf(masked).all():
            break

        min_idx = np.unravel_index(np.argmin(masked), masked.shape)
        i, j = min_idx

        qty = min(s[i], d[j])
        allocation[i, j] = int(qty)
        s[i] -= qty
        d[j] -= qty

        if s[i] == 0 and d[j] == 0:
            # Check remaining counts to avoid premature termination
            remaining_rows = sum(row_active)
            remaining_cols = sum(col_active)
            if remaining_rows > remaining_cols:
                row_active[i] = False
            else:
                col_active[j] = False
        elif s[i] == 0:
            row_active[i] = False
        else:
            col_active[j] = False

    return allocation, calculate_cost(allocation, costs)


def solve_vam(supply, demand, costs):
    """Solves initial BFS using Vogel's Approximation Method (VAM)."""
    s = np.array(supply, dtype=float).copy()
    d = np.array(demand, dtype=float).copy()
    c = np.array(costs, dtype=float).copy()
    m, n = c.shape
    allocation = np.zeros((m, n), dtype=int)

    row_active = [True] * m
    col_active = [True] * n

    while any(row_active) and any(col_active):
        active_rows = [i for i in range(m) if row_active[i]]
        active_cols = [j for j in range(n) if col_active[j]]

        # If only 1 row or 1 col remains, allocate all directly by least cost
        if len(active_rows) == 1:
            i = active_rows[0]
            for j in sorted(active_cols, key=lambda col: c[i, col]):
                if d[j] > 0 and s[i] > 0:
                    qty = min(s[i], d[j])
                    allocation[i, j] = int(qty)
                    s[i] -= qty
                    d[j] -= qty
            break

        if len(active_cols) == 1:
            j = active_cols[0]
            for i in sorted(active_rows, key=lambda row: c[row, j]):
                if s[i] > 0 and d[j] > 0:
                    qty = min(s[i], d[j])
                    allocation[i, j] = int(qty)
                    s[i] -= qty
                    d[j] -= qty
            break

        # Calculate row penalties
        row_penalties = {}
        for i in active_rows:
            active_vals = sorted([c[i, j] for j in active_cols])
            if len(active_vals) >= 2:
                row_penalties[i] = active_vals[1] - active_vals[0]
            elif len(active_vals) == 1:
                row_penalties[i] = active_vals[0]

        # Calculate col penalties
        col_penalties = {}
        for j in active_cols:
            active_vals = sorted([c[i, j] for i in active_rows])
            if len(active_vals) >= 2:
                col_penalties[j] = active_vals[1] - active_vals[0]
            elif len(active_vals) == 1:
                col_penalties[j] = active_vals[0]

        max_row_pen = max(row_penalties.values()) if row_penalties else -1
        max_col_pen = max(col_penalties.values()) if col_penalties else -1

        if max_row_pen >= max_col_pen:
            # Pick row with largest penalty
            candidates = [r for r, p in row_penalties.items() if p == max_row_pen]
            # Tie-break: lowest cost in candidates
            best_row = candidates[0]
            best_col = min(active_cols, key=lambda col: c[best_row, col])
            target_i, target_j = best_row, best_col
        else:
            # Pick col with largest penalty
            candidates = [col for col, p in col_penalties.items() if p == max_col_pen]
            best_col = candidates[0]
            best_row = min(active_rows, key=lambda row: c[row, best_col])
            target_i, target_j = best_row, best_col

        qty = min(s[target_i], d[target_j])
        allocation[target_i, target_j] = int(qty)
        s[target_i] -= qty
        d[target_j] -= qty

        if s[target_i] == 0 and d[target_j] == 0:
            if len(active_rows) > len(active_cols):
                row_active[target_i] = False
            else:
                col_active[target_j] = False
        elif s[target_i] == 0:
            row_active[target_i] = False
        else:
            col_active[target_j] = False

    return allocation, calculate_cost(allocation, costs)


def solve_lp_optimal(supply, demand, costs):
    """Solves the balanced Transportation Problem using SciPy Linear Programming (HiGHS)."""
    m, n = costs.shape
    num_vars = m * n

    # Flatten cost matrix for objective function: Minimize c^T x
    c = costs.flatten().astype(float)

    # Constraints:
    # 1. Supply constraints: sum_j x_ij = s_i for each i (m equations)
    # 2. Demand constraints: sum_i x_ij = d_j for each j (n equations)
    num_constraints = m + n
    A_eq = np.zeros((num_constraints, num_vars))
    b_eq = np.zeros(num_constraints)

    # Fill supply equations
    for i in range(m):
        for j in range(n):
            A_eq[i, i * n + j] = 1.0
        b_eq[i] = supply[i]

    # Fill demand equations
    for j in range(n):
        for i in range(m):
            A_eq[m + j, i * n + j] = 1.0
        b_eq[m + j] = demand[j]

    bounds = [(0, None) for _ in range(num_vars)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        allocation = np.round(res.x.reshape((m, n))).astype(int)
        opt_cost = int(round(res.fun))
        return allocation, opt_cost
    else:
        raise ValueError(f"LP Optimization failed: {res.message}")
