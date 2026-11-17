"""CPM / PERT analysis - Food Delivery App development project"""
import math
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# activity: (name, optimistic a, most likely m, pessimistic b, predecessors)
acts = {
 "A": ("Requirement gathering", 2, 3, 4, []),
 "B": ("UI/UX design",          3, 4, 5, ["A"]),
 "C": ("Database design",       2, 3, 4, ["A"]),
 "D": ("Backend development",   6, 8, 12, ["C"]),
 "E": ("Frontend development",  5, 7, 9, ["B"]),
 "F": ("API integration",       3, 4, 5, ["D", "E"]),
 "G": ("Payment gateway",       2, 3, 4, ["D"]),
 "H": ("Testing",               3, 4, 7, ["F", "G"]),
 "I": ("Deployment",            1, 2, 3, ["H"]),
}

def run(durations=None):
    te = {k: (v[1] + 4*v[2] + v[3]) / 6 for k, v in acts.items()}
    if durations: te.update(durations)
    order = list(acts)  # already topological
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
    return te, ES, EF, LS, LF, slack, crit, T

te, ES, EF, LS, LF, slack, crit, T = run()
var = {k: ((v[3]-v[1])/6)**2 for k, v in acts.items()}
print(f"{'Act':3} {'te':>6} {'ES':>6} {'EF':>6} {'LS':>6} {'LF':>6} {'Slack':>6} {'Var':>6}")
for k in acts:
    print(f"{k:3} {te[k]:6.2f} {ES[k]:6.2f} {EF[k]:6.2f} {LS[k]:6.2f} {LF[k]:6.2f} {slack[k]:6.2f} {var[k]:6.2f}")
print("Critical path:", " -> ".join(crit), "| Expected duration:", round(T, 2))
sd = math.sqrt(sum(var[k] for k in crit))
print("Std dev:", round(sd, 3))
for D in (24, 26, 28):
    print(f"P(finish <= {D} days) = {norm.cdf((D-T)/sd):.1%}")

# delay analysis
print("\nDelay analysis (new project duration)")
for k, d in [("D", 2), ("E", 2), ("G", 2), ("H", 2)]:
    _, *_, newT = run({k: te[k] + d})
    print(f"  {acts[k][0]} delayed by {d} days -> {newT:.2f} days (+{newT-T:.2f})")

# network diagram
G = nx.DiGraph()
G.add_node("Start"); G.add_node("End")
for k, v in acts.items():
    G.add_node(k)
    for p in v[4]: G.add_edge(p, k)
    if not v[4]: G.add_edge("Start", k)
G.add_edge("I", "End")
pos = {"Start": (0, 0), "A": (1, 0), "B": (2.2, 1), "C": (2.2, -1), "D": (3.4, -1), "E": (3.4, 1),
       "G": (4.6, -2), "F": (4.6, 0), "H": (5.8, 0), "I": (7, 0), "End": (8, 0)}
cedges = set(zip(["Start"] + crit, crit + ["End"]))
fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=["#D7263D" if e in cedges else "#9AA5B1" for e in G.edges],
                       width=[3.2 if e in cedges else 1.6 for e in G.edges], arrows=True, arrowsize=18,
                       node_size=1500, connectionstyle="arc3,rad=0")
cols = ["#D7263D" if n in crit else ("#1F2A44" if n in ("Start", "End") else "#3A6EA5") for n in G.nodes]
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cols, node_size=1500)
nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_weight="bold", font_size=13)
for k in acts:
    x, y = pos[k]
    ax.text(x, y - 0.42, f"{te[k]:.1f}d", ha="center", fontsize=10, color="#333")
ax.axis("off"); ax.set_xlim(-0.5, 8.5); ax.set_ylim(-2.6, 1.6)
plt.tight_layout(); plt.savefig("network.png", transparent=False, facecolor="white"); print("saved network.png")