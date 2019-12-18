import math
from termcolor import colored
import collections

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import colors

# parameters
# coupling conductance
gsd = 2
# leaky conductance
gl = 0.1
# synaptic conductances (soma)
ge = 0
gi = 0

Isyn_dnd = 0
# reversal potentials
Ee = 4.667
Ei = 0.333

phi_max = 150.0
k = 0.5
beta = 5
delta = 1

g_tot = gl + gsd

rate_list = [10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1, 10]

step = list()
weight_list = list()
order_of_magnitude = dict()

for i in range(-5, 3):
    step.append(10 ** i)

for s in step:
    for i in range(1, 10):
        weight_list.append(i * s)
        order_of_magnitude[i * s] = s

rates_fixed = dict()
weights_fixed = dict()

for i in rate_list:
    rates_fixed[i] = list()

for i in weight_list:
    weights_fixed[i] = list()

for weight in weight_list:

    for rate in rate_list:

        print("\n\n")
        if weight < 10 ** (-3):
            print(colored("32 bits weights only!", "red"))
        elif weight < 10 ** (-2):
            print(colored("weights can fit on 16 bits, but integer part will be 5 bits only", "yellow"))
        print("Weight: " + str(weight) + " Rate: " + str(rate))

        input_rate = rate

        Isyn_dnd = weight * rate

        V = Isyn_dnd * 10

        U = float((gsd * V)) / g_tot

        V_prev = V
        U_prev = U

        rate = phi_max / (1 + k * math.exp(beta * (delta - U)))

        print(" U: " + str(U) + " V: " + str(V) + " rate: " + str(rate))

        rates_fixed[input_rate].append((U, rate))
        weights_fixed[weight].append((U, rate))

print("\n\n\n")

for i in weights_fixed.keys():
    print("\n")
    for j in range(len(rate_list)):
        print("Weight: " + str(i) + " input rate:" + str(rate_list[j]) + "\tU: " + str(weights_fixed[i][j][0]) + "\trate: " + str(weights_fixed[i][j][1]))


print("\n\n\n")

for i in rates_fixed.keys():
    print("\n")
    for j in range(len(weight_list)):
        print("Rate: " + str(i) + " input weight:" + str(weight_list[j]) + " \tU: " + str(rates_fixed[i][j][0]) + " \trate: " + str(rates_fixed[i][j][1]))

# Plot stuff

x = list()
y = list()
z = list()
zu = list()
zv = list()
i_val  =list()

ordered_rates = collections.OrderedDict(sorted(rates_fixed.items(), reverse=True))

for i in ordered_rates.keys():

    j = 0

    x_row = list()
    y_row = list()
    z_row = list()
    zu_row = list()
    zv_row = list()
    i_row = list()

    while j < (len(weight_list) - 1):
        sum_rates_diff = 0
        sum_u_diff = 0
        sum_v_diff = 0
        sum_i_diff = 0
        cnt = 0
        while (j + 1) < len(weight_list) and order_of_magnitude[weight_list[j]] == order_of_magnitude[weight_list[j+1]]:
            sum_rates_diff += (ordered_rates[i][j+1][1] - ordered_rates[i][j][1])
            sum_u_diff += (ordered_rates[i][j+1][0] - ordered_rates[i][j][0])
            sum_v_diff += ((float(g_tot * ordered_rates[i][j + 1][0]) / gsd) - (float(g_tot * ordered_rates[i][j][0]) / gsd))
            sum_i_diff += ((float(g_tot * ordered_rates[i][j + 1][0]) / (10 * gsd)) - (float(g_tot * ordered_rates[i][j][0]) / (10 * gsd)))
            j += 1
            cnt += 1
        if cnt > 0:
            y_row.append(order_of_magnitude[weight_list[j]])
            z_row.append(float(sum_rates_diff) / cnt)
            zu_row.append(float(sum_u_diff) / cnt)
            zv_row.append(float(sum_v_diff) / cnt)
            i_row.append(float(sum_i_diff) / cnt)
        j += 1

        # i = in rate, order_of_magnitude[weight_list[j]] = order of magnitude of in weight, rates_fixed[i][j+1][1] - rates_fixed[i][j][1] = rate difference

    x.append(i)
    y.append(y_row)
    z.append(z_row)
    zu.append(zu_row)
    zv.append(zv_row)
    i_val.append(i_row)

cbar_ticks = [10 ** (-8), 10 ** (-7), 10 ** (-6), 10 **(-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 1, 10, 100]

f1 = plt.figure(1)
ax = sns.heatmap(pd.DataFrame(z, x, columns=y_row), annot=True, cbar=True, linewidths=.5, cbar_kws={"ticks": cbar_ticks}, norm=colors.SymLogNorm(linthresh=0.00000001, linscale=0.00000001, vmin=10 ** (-8), vmax=100))
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.xlabel("Input Weight")
plt.ylabel("Input Rate")
plt.title("Changes in Output Rate")

f2 = plt.figure(2)
ax = sns.heatmap(pd.DataFrame(zu, x, columns=y_row), annot=True, cbar=True, linewidths=.5, cbar_kws={"ticks": cbar_ticks}, norm=colors.SymLogNorm(linthresh=0.00000001, linscale=0.00000001, vmin=10 ** (-8), vmax=100))
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.xlabel("Input Weight")
plt.ylabel("Input Rate")
plt.title("Changes in Output Somatic Voltage")

f3 = plt.figure(3)
ax = sns.heatmap(pd.DataFrame(zv, x, columns=y_row), annot=True, cbar=True, linewidths=.5, cbar_kws={"ticks": cbar_ticks}, norm=colors.SymLogNorm(linthresh=0.00000001, linscale=0.00000001, vmin=10 ** (-8), vmax=100))
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.xlabel("Input Weight")
plt.ylabel("Input Rate")
plt.title("Changes in Output Dendritic Voltage")

f4 = plt.figure(4)
ax = sns.heatmap(pd.DataFrame(i_val, x, columns=y_row), annot=True, cbar=True, linewidths=.5, cbar_kws={"ticks": cbar_ticks}, norm=colors.SymLogNorm(linthresh=0.00000001, linscale=0.00000001, vmin=10 ** (-8), vmax=100))
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.xlabel("Input Weight")
plt.ylabel("Input Rate")
plt.title("Changes in Current")
plt.show()
