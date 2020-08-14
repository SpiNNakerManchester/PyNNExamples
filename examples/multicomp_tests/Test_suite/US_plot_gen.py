import matplotlib.pyplot as plt
import numpy as np

Um = []
U = []
V = []

xsin = np.linspace(0, np.pi, 50)
xpar = np.linspace(-28, 27, 55)
par = [0.0001754 * (i ** 2) for i in xpar]
par.insert(28, 0.0)
sin = [np.sin(i) for i in xsin]

exc_som_val = [par[-1]]
exc_som_val.extend(sin[3:47])
exc_som_val.extend(par[:-1])

soma_vals = [exc_som_val[i % len(exc_som_val)] for i in range(22000)]
soma_inh_vals = [2 for i in range(22000)]

soma_vals[:1000] = [0 for _ in range(1000)]
soma_inh_vals[:1000] = [0 for _ in range(1000)]
soma_vals[20000:] = [0 for _ in range(2000)]
soma_inh_vals[20000:] = [0 for _ in range(2000)]


with open("/localhome/g90604lp/um.txt", "r") as fp:

    l = fp.readline()
    while l != "":
        Um.append(float(l))
        l = fp.readline()

with open("/localhome/g90604lp/u.txt", "r") as fp:

    l = fp.readline()
    while l != "":
        U.append(float(l))
        l = fp.readline()

with open("/localhome/g90604lp/v.txt", "r") as fp:

    l = fp.readline()
    while l != "":
        V.append(float(l))
        l = fp.readline()

x = [_ for _ in range(400)]

Um_plot = Um[19800:20200]
#Um_plot.extend(Um[19800:20200])

U_plot = U[19800:20200]
#U_plot.extend(U[19800:20200])

V_plot = V[19800:20200]
#V_plot.extend(V[19800:20200])

#plt.plot(x[0:200], Um_plot[0:200], color="red", linewidth=2)
#plt.plot(x[200:400], Um_plot[200:400], "--", color="red", linewidth=2)
plt.plot(x, soma_vals[19800:20200], color="blue", linewidth=2)
plt.plot(x, soma_inh_vals[19800:20200], color="red", linewidth=2)
#plt.plot(x, U, "--", color="aqua", linewidth=1.5, label="U expected")
#plt.plot(x, V, "--", color="lightgreen", linewidth=1.5, label="V expected")
plt.title("Urbanczik-Senn plasticity Voltages")
plt.yticks([1, 2], fontsize=20)
plt.xticks([0, 200, 400], [19.8, 20, 20.2], fontsize=20)
plt.show()