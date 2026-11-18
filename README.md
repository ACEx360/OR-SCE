# Transportation Cost Minimization System
Operations Research — SCE Project

**Students:**
- Alpesh Thool 22310727 05
- chinmay kale 22310757 21
- Tayyab Mohammed 22311292 32
- Tanishq Raina 22310084 17


**Class / Division:** B.Tech Data Science  
**Faculty Guide:** Prof. VijayKumar Ghule  

---

## Project Overview
This project is an Operations Research decision-support application designed to solve and optimize the classical **Transportation Problem**, minimizing overall logistics shipping costs from multiple supply origins (factories) to multiple demand destinations (warehouses).

It also includes a companion **CPM / PERT Project Scheduling** module for milestone planning, critical path determination, and project duration risk analysis.

---

## Solvers and Methods Compared
1. **Northwest Corner Rule (NWC)**: Classical top-left heuristic allocation.
2. **Least Cost Method (LCM / Matrix Minimum)**: Greedy unit-cost minimization.
3. **Vogel’s Approximation Method (VAM)**: Penalty-based heuristic for near-optimal initial feasible solutions.
4. **LP Optimal Benchmark (SciPy HiGHS)**: Linear programming global minimum cost benchmark.

### Benchmark Results (3x3 Problem from Presentation Slides 7–9)
| Method | Total Cost | Optimality Gap | Status |
| :--- | :---: | :---: | :---: |
| **Northwest Corner Rule** | 910 | +10 | Initial Feasible Solution |
| **Least Cost Method** | 900 | 0 | Matches Optimal |
| **Vogel’s Approximation (VAM)** | 900 | 0 | Matches Optimal |
| **SciPy LP Optimal Benchmark** | 900 | 0 | Globally Optimal Minimum |

#### Optimal Allocation Breakdown (Slide 9):
- Factory 1 -> Warehouse A: 70 x 4 = 280
- Factory 2 -> Warehouse B: 90 x 3 = 270
- Factory 3 -> Warehouse A: 10 x 6 = 60
- Factory 3 -> Warehouse B: 10 x 5 = 50
- Factory 3 -> Warehouse C: 60 x 4 = 240
- **Total Minimum Cost**: 280 + 270 + 60 + 50 + 240 = **900 cost units**

---

## Running the Application

### 1. Launch Web Dashboard
```bash
streamlit run app.py
```
Or via python launcher:
```bash
python main.py --ui
```

### 2. Run Terminal CLI Benchmark
```bash
python main.py
```

---

## Tech Stack
- Python 3.13
- Streamlit
- SciPy (`scipy.optimize.linprog`)
- NumPy and Pandas
- Plotly
- NetworkX and Matplotlib
