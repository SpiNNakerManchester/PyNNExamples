import glob
import matplotlib.pyplot as plt
import numpy as np

values = {"dma_read": [],
          "state_update": [],
          "loop_time": [],
          "state_no_loop": [],
          "total": []}

for txtfile in glob.iglob("/localhome/g90604lp/ICPP_res/neurons/*.txt"):
    with open(txtfile, "r") as fp:
        l1 = fp.readline()
        l2 = fp.readline()
        l3 = fp.readline()
        x_axis = []
        x = 2
        k = 2
        while k < 15:

            fp.readline()
            dma_values = l1.split("\n")[0].split(" ")
            dma_sum = 0
            for v in range(len(dma_values)):
                dma_sum += (int(dma_values[v]) * 0.005)
            dma_avg = float(dma_sum) / len(dma_values)
            values["dma_read"].append(dma_avg)

            state_values = l2.split(" ")
            state_sum = 0
            #for v in range(len(state_values)):
            #    state_sum += (int(state_values[v]) * 0.005)
            #state_avg = float(state_sum) / len(state_values)
            state_avg = int(state_values[0]) * 0.005
            values["state_update"].append(state_avg)

            values["total"].append(state_avg + dma_avg)

            loop_time = l3[:-1]
            values["loop_time"].append(float(loop_time) * 64)
            values["state_no_loop"].append(state_avg - (float(loop_time) * 64))


            l1 = fp.readline()
            l2 = fp.readline()
            l3 = fp.readline()
            x_axis.append(x)
            x += 1
            k += 1
    name = txtfile.split("/")[5].split(".")[0]

    plt.plot(x_axis, values["total"], "o-", color = "blue")
    #plt.plot(x_axis, values["state_update"], "o-", color="magenta")
    plt.plot(x_axis, values["state_no_loop"], "o-")
    plt.plot(x_axis, values["loop_time"], "o-", color="purple")
    plt.plot(x_axis, values["dma_read"], "o-", color="lightskyblue")

    plt.legend(["Total", 'Neuron update time', 'Synaptic summation loop time', 'DMA timings'], prop={'size': 16})
    plt.grid()
    plt.yticks(np.arange(0, 105, step=5),)
    plt.xticks(np.arange(2, 15, step=1), size=17)
    #plt.title()
    #plt.yscale("log")
    plt.xlabel("Connected synapse cores", size=28)
    plt.ylabel("Time (microseconds)", size=28)
    plt.show()

    values["dma_read"] = []
    values["state_update"] = []
    values["loop_time"] = []
    values["state_no_loop"] = []
    values["total"] = []
