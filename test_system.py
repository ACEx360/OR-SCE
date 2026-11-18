"""
Sanity check and automated test suite for Transportation Cost Minimization System.
Tests:
1. PPT benchmark values (NWC=910, LCM=900, VAM=900, LP=900).
2. Constraint satisfaction (Row sums = Supply, Col sums = Demand).
3. Non-negativity constraint (all x_ij >= 0).
4. Edge cases (varying dimensions like 2x3, 4x3).
5. Breakdown calculation verification.
"""

import numpy as np
from transportation_solver import (
    solve_northwest_corner,
    solve_least_cost,
    solve_vam,
    solve_lp_optimal,
    format_breakdown,
)


def verify_feasibility(alloc, supply, demand, name):
    # Non-negativity
    assert (alloc >= 0).all(), f"[{name}] Negative allocation found!"
    # Supply constraints
    shipped = np.sum(alloc, axis=1)
    assert np.array_equal(shipped, supply), f"[{name}] Supply violated: shipped {shipped} != {supply}"
    # Demand constraints
    received = np.sum(alloc, axis=0)
    assert np.array_equal(received, demand), f"[{name}] Demand violated: received {received} != {demand}"


def test_ppt_benchmark():
    print("Testing Slide 7-9 Benchmark Data...")
    supply = np.array([70, 90, 80])
    demand = np.array([80, 100, 60])
    costs = np.array([
        [4, 6, 8],
        [5, 3, 7],
        [6, 5, 4]
    ])

    alloc_nwc, cost_nwc = solve_northwest_corner(supply, demand, costs)
    alloc_lcm, cost_lcm = solve_least_cost(supply, demand, costs)
    alloc_vam, cost_vam = solve_vam(supply, demand, costs)
    alloc_lp, cost_lp = solve_lp_optimal(supply, demand, costs)

    # Feasibility
    verify_feasibility(alloc_nwc, supply, demand, "NWC")
    verify_feasibility(alloc_lcm, supply, demand, "LCM")
    verify_feasibility(alloc_vam, supply, demand, "VAM")
    verify_feasibility(alloc_lp, supply, demand, "LP Optimal")

    # Exact cost checks from presentation
    assert cost_nwc == 910, f"NWC cost expected 910, got {cost_nwc}"
    assert cost_lcm == 900, f"LCM cost expected 900, got {cost_lcm}"
    assert cost_vam == 900, f"VAM cost expected 900, got {cost_vam}"
    assert cost_lp == 900, f"LP cost expected 900, got {cost_lp}"

    # Breakdown verification
    shipments, calc_str, total = format_breakdown(alloc_lp, costs)
    assert total == 900, f"Breakdown total expected 900, got {total}"
    assert len(shipments) == 5, f"Expected 5 non-zero shipments, got {len(shipments)}"
    print("  ✓ PPT benchmark verified: NWC=910, LCM=900, VAM=900, LP=900")


def test_custom_dimensions():
    print("\nTesting Custom Dimensions & Variations...")
    # 2 sources, 4 destinations
    supply_2x4 = np.array([120, 80])
    demand_2x4 = np.array([50, 60, 40, 50])
    costs_2x4 = np.array([
        [3, 5, 7, 6],
        [2, 4, 8, 3]
    ])

    for solver, name in [
        (solve_northwest_corner, "NWC"),
        (solve_least_cost, "LCM"),
        (solve_vam, "VAM"),
        (solve_lp_optimal, "LP"),
    ]:
        alloc, cost = solver(supply_2x4, demand_2x4, costs_2x4)
        verify_feasibility(alloc, supply_2x4, demand_2x4, f"2x4 {name}")
    print("  ✓ 2x4 problem feasibility verified across all 4 solvers")

    # 4 sources, 3 destinations
    supply_4x3 = np.array([50, 70, 80, 40])
    demand_4x3 = np.array([100, 60, 80])
    costs_4x3 = np.array([
        [8, 4, 6],
        [5, 9, 3],
        [7, 2, 5],
        [4, 6, 8]
    ])

    for solver, name in [
        (solve_northwest_corner, "NWC"),
        (solve_least_cost, "LCM"),
        (solve_vam, "VAM"),
        (solve_lp_optimal, "LP"),
    ]:
        alloc, cost = solver(supply_4x3, demand_4x3, costs_4x3)
        verify_feasibility(alloc, supply_4x3, demand_4x3, f"4x3 {name}")
    print("  ✓ 4x3 problem feasibility verified across all 4 solvers")


if __name__ == "__main__":
    test_ppt_benchmark()
    test_custom_dimensions()
    print("\n" + "=" * 50)
    print("ALL SANITY CHECKS & TESTS PASSED!")
    print("=" * 50)
