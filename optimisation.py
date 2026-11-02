"""
Part 3: Optimisation
Furniture manufacturer - maximise profit from tables and chairs
Table: profit 6, 30 bf wood, 5 hrs labour
Chair: profit 8, 20 bf wood, 10 hrs labour
Available: 300 bf wood, 110 hrs labour
"""

import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMaximize, LpVariable, LpStatus, value, PULP_CBC_CMD

# =================================================================
# Q1: Integer Linear Programming formulation and PuLP solution
# =================================================================
prob = LpProblem('Furniture', LpMaximize)

T = LpVariable('Tables', lowBound=0, cat='Integer')
C = LpVariable('Chairs', lowBound=0, cat='Integer')

prob += 6 * T + 8 * C, 'Profit'
prob += 30 * T + 20 * C <= 300, 'Wood'
prob += 5 * T + 10 * C <= 110, 'Labour'

prob.solve(PULP_CBC_CMD(msg=0))

opt_T = T.varValue
opt_C = C.varValue
opt_profit = value(prob.objective)
wood_used = 30 * opt_T + 20 * opt_C
labour_used = 5 * opt_T + 10 * opt_C

print("Status:", LpStatus[prob.status])
print("Tables:", opt_T)
print("Chairs:", opt_C)
print("Profit:", opt_profit)
print("Wood used:", wood_used)
print("Labour used:", labour_used)

# Feasible region plot (Figure 1)
fig, ax = plt.subplots(figsize=(7, 6))

Tline = np.linspace(0, 12, 400)
wood_C = np.clip((300 - 30 * Tline) / 20, 0, None)
labour_C = np.clip((110 - 5 * Tline) / 10, 0, None)

ax.plot(Tline, wood_C, label='Wood: 30T + 20C = 300', color='#4C72B0')
ax.plot(Tline, labour_C, label='Labour: 5T + 10C = 110', color='#DD8452')

feasible_upper = np.minimum(wood_C, labour_C)
ax.fill_between(Tline, 0, feasible_upper, where=(Tline <= 10),
                 color='#C5D9F1', alpha=0.6, label='Feasible region')

corners = [(0, 0), (10, 0), (opt_T, opt_C), (0, 11)]
ax.scatter([c[0] for c in corners], [c[1] for c in corners], color='black', zorder=5)
for x, yv in corners:
    ax.annotate(f'({x:.0f},{yv:.0f})', (x, yv), textcoords="offset points",
                xytext=(6, 6), fontsize=9)

ax.scatter([opt_T], [opt_C], color='red', s=100, zorder=6,
           label=f'Optimal ({opt_T:.0f}, {opt_C:.0f})')

Tobj = np.linspace(0, 16, 10)
Cobj = (opt_profit - 6 * Tobj) / 8
ax.plot(Tobj, Cobj, linestyle='--', color='green',
        label=f'Profit = {opt_profit:.0f} (objective line)')

ax.set_xlim(0, 16)
ax.set_ylim(0, 16)
ax.set_xlabel('Tables (T)')
ax.set_ylabel('Chairs (C)')
ax.set_title('Feasible Region and Optimal Solution')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig('results/feasible_region.png', dpi=140)
plt.show()

