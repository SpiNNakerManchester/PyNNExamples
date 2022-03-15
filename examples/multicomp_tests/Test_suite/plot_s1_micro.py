import matplotlib.pyplot as plt
import numpy as np

va = "/localhome/g90604lp/selfp/va.txt"
vb = "/localhome/g90604lp/selfp/vb.txt"
usst = "/localhome/g90604lp/selfp/usst.txt"
utop = "/localhome/g90604lp/selfp/utop.txt"

va_list = []
vb_list = []
usst_list = []
utop_list = []

with open(va, "r") as fp:
    line = fp.read()
    vals = line.split("uS")
    for v in vals:
        nums = v.split(" ")
        nums = [float(n.strip("[]\n")) for n in nums if (n != '' and n != '[' and n != ']')]
        va_list.append(nums)

with open(vb, "r") as fp:
    line = fp.read()
    vals = line.split("uS")
    for v in vals:
        nums = v.split(" ")
        nums = [float(n.strip("[]\n")) for n in nums if (n != '' and n != '[' and n != ']')]
        vb_list.append(nums)

with open(usst, "r") as fp:
    line = fp.read()
    vals = line.split("mV")
    for v in vals:
        nums = v.split(" ")
        nums = [float(n.strip("[]\n")) for n in nums if (n != '' and n != '[' and n != ']')]
        usst_list.append(nums)

with open(utop, "r") as fp:
    line = fp.read()
    vals = line.split("mV")
    for v in vals:
        nums = v.split(" ")
        nums = [float(n.strip("[]\n")) for n in nums if (n != '' and n != '[' and n != ']')]
        utop_list.append(nums)

va_low = [va_list[i][9] for i in range(1000)]
vb_low = [vb_list[i][9] for i in range(1000)]
usst_low = [usst_list[i][4] for i in range(1000)]
utop_low = [utop_list[i][4] for i in range(1000)]

va_high = [va_list[i][6] for i in range(50001, 51001)]
vb_high = [vb_list[i][9] for i in range(50001, 51001)]
usst_high = [usst_list[i][9] for i in range(50001, 51001)]
utop_high = [utop_list[i][9] for i in range(50001, 51001)]


plt.rcParams.update({'font.size': 20})
fix, ax = plt.subplots(3, 1)

ax[0].plot([i for i in range(1000)], utop_low, label="Top-down feedback")
ax[0].plot([i for i in range(1000)], usst_low, label="Lateral cancellation")

ax[0].legend(prop={'size': 16.5})


ax[1]. plot([i for i in range(1000)], vb_low, label="Sensory input")

ax[1].legend(prop={'size': 16.5})

ax[2]. plot([i for i in range(1000)], va_low)
ax[2]. plot([i for i in range(1000)], [0 for _ in range(1000)], "--", label="Apical potential")
ax[2].legend(prop={'size': 13.5})

ax[2].set_xlabel("Time")
ax[0].set_ylim([-10, 10])

plt.legend(prop={'size': 16.5})
plt.show()

#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, N)))
fix, ax = plt.subplots(3, 1)

ax[0].plot([i for i in range(1000)], utop_high, label="Top-down feedback")
ax[0].plot([i for i in range(1000)], usst_high, label="Lateral cancellation")

ax[0].legend(prop={'size': 16.5})


ax[1]. plot([i for i in range(1000)], vb_high, label="Sensory input")

ax[1].legend(prop={'size': 16.5})

ax[2]. plot([i for i in range(1000)], va_high)
ax[2]. plot([i for i in range(1000)], [0 for _ in range(1000)], "--", label="Apical potential")
ax[2].legend(prop={'size': 14})

ax[2].set_xlabel("Time")

plt.legend(prop={'size': 16.5})
plt.show()

