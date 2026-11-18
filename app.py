"""
Transportation Cost Minimization System
Operations Research - SCE Project
Students: Alpesh Thool (05), Mohammad Tayyab (31)
Faculty Guide: Prof. VijayKumar Ghule
"""

import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

from transportation_solver import (
    format_breakdown,
    solve_least_cost,
    solve_lp_optimal,
    solve_northwest_corner,
    solve_vam,
)

st.set_page_config(
    page_title="Transportation Optimization System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("Transportation Cost Minimization System")
st.caption(
    "Operations Research SCE Project | B.Tech Data Science | "
    "Alpesh Thool (05), Mohammad Tayyab (31) | Guide: Prof. VijayKumar Ghule"
)
st.divider()

tab_transport, tab_cpm = st.tabs([
    "Transportation Model",
    "CPM / PERT Scheduling",
])

# -----------------------------------------------------------------------------
# TAB 1: TRANSPORTATION MODEL
# -----------------------------------------------------------------------------
with tab_transport:
    st.sidebar.subheader("Configuration")

    dataset_choice = st.sidebar.radio(
        "Dataset Source",
        ["Presentation Benchmark (3x3)", "Custom Problem"],
        index=0,
    )

    if dataset_choice == "Presentation Benchmark (3x3)":
        num_factories = 3
        num_warehouses = 3
        default_supply = [70, 90, 80]
        default_demand = [80, 100, 60]
        default_costs = [
            [4, 6, 8],
            [5, 3, 7],
            [6, 5, 4],
        ]
        factory_names = ["Factory 1", "Factory 2", "Factory 3"]
        warehouse_names = ["Warehouse A", "Warehouse B", "Warehouse C"]
    else:
        num_factories = st.sidebar.slider("Number of Sources (Factories)", 2, 6, 3)
        num_warehouses = st.sidebar.slider("Number of Destinations (Warehouses)", 2, 6, 3)
        default_supply = [100] * num_factories
        default_demand = [int(sum(default_supply) / num_warehouses)] * num_warehouses
        default_demand[-1] = sum(default_supply) - sum(default_demand[:-1])
        default_costs = [[np.random.randint(2, 10) for _ in range(num_warehouses)] for _ in range(num_factories)]
        factory_names = [f"Factory {i+1}" for i in range(num_factories)]
        warehouse_names = [f"Warehouse {chr(65+j)}" for j in range(num_warehouses)]

    st.subheader("Input Matrix and Capacities")
    st.write("Unit shipping costs ($c_{ij}$), factory supply limits ($s_i$), and warehouse demands ($d_j$):")

    col_cost, col_sup, col_dem = st.columns([3, 1, 1])

    with col_cost:
        st.markdown("**Unit Cost Matrix (c_ij)**")
        cost_df = pd.DataFrame(default_costs, index=factory_names, columns=warehouse_names)
        edited_cost_df = st.data_editor(cost_df, use_container_width=True, key="cost_matrix")

    with col_sup:
        st.markdown("**Factory Supply (s_i)**")
        supply_df = pd.DataFrame({"Supply": default_supply}, index=factory_names)
        edited_supply_df = st.data_editor(supply_df, use_container_width=True, key="supply_vec")

    with col_dem:
        st.markdown("**Warehouse Demand (d_j)**")
        demand_df = pd.DataFrame({"Demand": default_demand}, index=warehouse_names)
        edited_demand_df = st.data_editor(demand_df, use_container_width=True, key="demand_vec")

    costs = edited_cost_df.to_numpy(dtype=float)
    supply = edited_supply_df["Supply"].to_numpy(dtype=float)
    demand = edited_demand_df["Demand"].to_numpy(dtype=float)

    total_supply = int(np.sum(supply))
    total_demand = int(np.sum(demand))
    is_balanced = total_supply == total_demand

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Total Supply", f"{total_supply} units")
    col_s2.metric("Total Demand", f"{total_demand} units")
    with col_s3:
        if is_balanced:
            st.success(f"Problem is balanced: Supply ({total_supply}) = Demand ({total_demand})")
        else:
            st.error(f"Problem is unbalanced: Supply ({total_supply}) != Demand ({total_demand})")

    if not is_balanced:
        st.warning("Please balance total supply and total demand before running optimization.")
    else:
        # Solve all methods
        alloc_nwc, cost_nwc = solve_northwest_corner(supply, demand, costs)
        alloc_lcm, cost_lcm = solve_least_cost(supply, demand, costs)
        alloc_vam, cost_vam = solve_vam(supply, demand, costs)
        alloc_lp, cost_lp = solve_lp_optimal(supply, demand, costs)

        st.subheader("Method Comparison")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LP Optimal Benchmark", f"{cost_lp}")
        m2.metric(
            "Northwest Corner",
            f"{cost_nwc}",
            delta=f"+{cost_nwc - cost_lp}" if cost_nwc > cost_lp else "0",
            delta_color="inverse",
        )
        m3.metric(
            "Least Cost Method",
            f"{cost_lcm}",
            delta=f"+{cost_lcm - cost_lp}" if cost_lcm > cost_lp else "0",
            delta_color="inverse",
        )
        m4.metric(
            "Vogel's Approximation (VAM)",
            f"{cost_vam}",
            delta=f"+{cost_vam - cost_lp}" if cost_vam > cost_lp else "0",
            delta_color="inverse",
        )

        # Plotly Comparison Chart
        comparison_df = pd.DataFrame({
            "Method": [
                "Northwest Corner",
                "Least Cost Method",
                "Vogel's Approximation (VAM)",
                "LP Optimal Benchmark",
            ],
            "Total Cost": [cost_nwc, cost_lcm, cost_vam, cost_lp],
            "Category": ["Heuristic", "Heuristic", "Heuristic", "Optimal"],
        })

        fig_bar = px.bar(
            comparison_df,
            x="Method",
            y="Total Cost",
            color="Category",
            color_discrete_map={"Heuristic": "#64748b", "Optimal": "#2563eb"},
            text="Total Cost",
            template="plotly_white",
            title="Comparison of Total Transportation Cost across Methods",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            yaxis_range=[0, max(comparison_df["Total Cost"]) * 1.15],
            showlegend=True,
            height=380,
            margin=dict(t=50, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Detailed Allocation Matrix & Step Verification
        st.subheader("Allocations and Route Verification")

        selected_method = st.radio(
            "Inspect Allocation Details:",
            [
                "LP Optimal Benchmark",
                "Vogel's Approximation (VAM)",
                "Least Cost Method",
                "Northwest Corner",
            ],
            horizontal=True,
        )

        if selected_method == "LP Optimal Benchmark":
            active_alloc = alloc_lp
            active_cost = cost_lp
            method_note = "Optimal allocation derived from linear programming formulation (SciPy HiGHS)."
        elif selected_method == "Vogel's Approximation (VAM)":
            active_alloc = alloc_vam
            active_cost = cost_vam
            method_note = "VAM penalizes high-cost alternate routes to find near-optimal initial solutions."
        elif selected_method == "Least Cost Method":
            active_alloc = alloc_lcm
            active_cost = cost_lcm
            method_note = "Greedy allocation starting from cells with lowest unit costs."
        else:
            active_alloc = alloc_nwc
            active_cost = cost_nwc
            method_note = "Top-left cell allocation following supply and demand exhaustion order."

        st.caption(method_note)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("**Allocation Matrix (Units Shipped)**")
            alloc_df = pd.DataFrame(active_alloc, index=factory_names, columns=warehouse_names)

            def style_alloc(val):
                if val > 0:
                    return "background-color: #f1f5f9; font-weight: 600; color: #0f172a;"
                return "color: #94a3b8;"

            st.dataframe(alloc_df.style.map(style_alloc), use_container_width=True)

            shipments, calc_str, _ = format_breakdown(active_alloc, costs, factory_names, warehouse_names)
            st.markdown("**Cost Calculation Breakdown**")
            st.code(calc_str, language="text")

        with col_right:
            st.markdown("**Feasibility Verification**")
            shipped = np.sum(active_alloc, axis=1)
            received = np.sum(active_alloc, axis=0)

            factory_check = pd.DataFrame({
                "Source": factory_names,
                "Shipped": shipped,
                "Limit": supply,
                "Status": ["Satisfied" if s == l else "Unmet" for s, l in zip(shipped, supply)],
            })
            st.dataframe(factory_check, use_container_width=True, hide_index=True)

            warehouse_check = pd.DataFrame({
                "Destination": warehouse_names,
                "Received": received,
                "Requirement": demand,
                "Status": ["Satisfied" if r == d else "Unmet" for r, d in zip(received, demand)],
            })
            st.dataframe(warehouse_check, use_container_width=True, hide_index=True)

        st.subheader("Route Flow Diagram")
        source_labels = list(factory_names)
        target_labels = list(warehouse_names)
        all_labels = source_labels + target_labels

        sankey_sources = []
        sankey_targets = []
        sankey_values = []
        sankey_hover = []

        for i in range(len(source_labels)):
            for j in range(len(target_labels)):
                qty = active_alloc[i, j]
                if qty > 0:
                    sankey_sources.append(i)
                    sankey_targets.append(len(source_labels) + j)
                    sankey_values.append(qty)
                    sankey_hover.append(
                        f"{source_labels[i]} -> {target_labels[j]}<br>"
                        f"Volume: {qty} units<br>"
                        f"Cost rate: ${costs[i, j]}/unit<br>"
                        f"Total: ${qty * costs[i, j]}"
                    )

        fig_sankey = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="#64748b", width=0.5),
                        label=all_labels,
                        color=["#334155"] * len(source_labels) + ["#2563eb"] * len(target_labels),
                    ),
                    link=dict(
                        source=sankey_sources,
                        target=sankey_targets,
                        value=sankey_values,
                        customdata=sankey_hover,
                        hovertemplate="%{customdata}<extra></extra>",
                        color="rgba(148, 163, 184, 0.45)",
                    ),
                )
            ]
        )
        fig_sankey.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(t=30, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_sankey, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: CPM / PERT SCHEDULING
# -----------------------------------------------------------------------------
with tab_cpm:
    st.subheader("CPM / PERT Project Scheduling")
    st.caption("Food Delivery App Development Case Study")

    acts = {
        "A": ("Requirement gathering", 2, 3, 4, []),
        "B": ("UI/UX design", 3, 4, 5, ["A"]),
        "C": ("Database design", 2, 3, 4, ["A"]),
        "D": ("Backend development", 6, 8, 12, ["C"]),
        "E": ("Frontend development", 5, 7, 9, ["B"]),
        "F": ("API integration", 3, 4, 5, ["D", "E"]),
        "G": ("Payment gateway", 2, 3, 4, ["D"]),
        "H": ("Testing", 3, 4, 7, ["F", "G"]),
        "I": ("Deployment", 1, 2, 3, ["H"]),
    }

    te = {k: (v[1] + 4 * v[2] + v[3]) / 6 for k, v in acts.items()}
    var = {k: ((v[3] - v[1]) / 6) ** 2 for k, v in acts.items()}
    order = list(acts)

    ES, EF = {}, {}
    for k in order:
        ES[k] = max([EF[p] for p in acts[k][4]], default=0)
        EF[k] = ES[k] + te[k]
    T = max(EF.values())

    succ = {k: [j for j in order if k in acts[j][4]] for k in order}
    LF, LS = {}, {}
    for k in reversed(order):
        LF[k] = min([LS[s] for s in succ[k]], default=T)
        LS[k] = LF[k] - te[k]

    slack = {k: LS[k] - ES[k] for k in order}
    crit = [k for k in order if abs(slack[k]) < 1e-9]
    sd = math.sqrt(sum(var[k] for k in crit))

    c_cpm1, c_cpm2, c_cpm3 = st.columns(3)
    c_cpm1.metric("Critical Path", " -> ".join(crit))
    c_cpm2.metric("Expected Duration", f"{T:.2f} days")
    c_cpm3.metric("Standard Deviation (sigma)", f"{sd:.2f} days")

    cpm_table = []
    for k in acts:
        cpm_table.append({
            "Activity": k,
            "Description": acts[k][0],
            "Optimistic (a)": acts[k][1],
            "Most Likely (m)": acts[k][2],
            "Pessimistic (b)": acts[k][3],
            "Expected te": round(te[k], 2),
            "ES": round(ES[k], 2),
            "EF": round(EF[k], 2),
            "LS": round(LS[k], 2),
            "LF": round(LF[k], 2),
            "Slack": round(slack[k], 2),
            "Critical": "Yes" if k in crit else "No",
        })

    st.markdown("**Activity Schedule and Slack Table**")
    st.dataframe(pd.DataFrame(cpm_table), use_container_width=True, hide_index=True)

    st.markdown("**Probability of Project Completion**")
    target_d = st.slider("Target Duration (Days)", 20, 32, 26)
    z_score = (target_d - T) / sd
    prob = norm.cdf(z_score)
    st.write(f"Probability of completion within **{target_d} days**: **{prob:.1%}** (Z = {z_score:.2f})")

    st.markdown("**Delay Sensitivity Analysis**")
    delay_act = st.selectbox(
        "Select activity to delay:",
        list(acts.keys()),
        format_func=lambda x: f"{x}: {acts[x][0]}",
    )
    delay_days = st.slider("Delay duration (days):", 0, 10, 2)
    new_te = te.copy()
    new_te[delay_act] += delay_days

    new_EF = {}
    for k in order:
        new_es = max([new_EF[p] for p in acts[k][4]], default=0)
        new_EF[k] = new_es + new_te[k]
    new_T = max(new_EF.values())
    st.info(
        f"Delaying {acts[delay_act][0]} by {delay_days} days changes project completion "
        f"from {T:.2f} to {new_T:.2f} days (+{new_T - T:.2f} days)."
    )
