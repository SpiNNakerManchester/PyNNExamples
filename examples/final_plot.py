import glob
import matplotlib.pyplot as plt

values = {"dma_read": [],
          "state_update": [],
          "loop_time": [],
          "state_no_loop": []}

for txtfile in glob.iglob("/localhome/g90604lp/ICPP_res/neurons/*.txt"):
    with open(txtfile, "r") as fp:
        l1 = fp.readline()
        l2 = fp.readline()
        l3 = fp.readline()
        x_axis = []
        x = 2
        while fp.readline() != '':

            dma_values = l1.split("\n")[0].split(" ")
            dma_sum = 0
            for v in range(len(dma_values)):
                dma_sum += (int(dma_values[v]) * 0.005)
            dma_avg = float(dma_sum) / len(dma_values)
            values["dma_read"].append(dma_avg)

            state_values = l2.split(" ")
            state_sum = 0
            for v in range(len(state_values)):
                state_sum += (int(state_values[v]) * 0.005)
            state_avg = float(state_sum) / len(state_values)
            values["state_update"].append(state_avg)

            loop_time = l3[:-1]
            values["loop_time"].append(float(loop_time))
            values["state_no_loop"].append(state_avg - float(loop_time))


            l1 = fp.readline()
            l2 = fp.readline()
            l3 = fp.readline()
            x_axis.append(x)
            x += 1
    plt.plot(x_axis, values["dma_read"])
    plt.plot(x_axis, values["state_update"])
    plt.plot(x_axis, values["loop_time"])
    plt.plot(x_axis, values["state_no_loop"])
    plt.legend(['DMA timings', 'synaptic contribution loop time', 'Neuron update time', 'Total neuron update time'])
    #plt.yscale("log")
    plt.show()

    values["dma_read"] = []
    values["state_update"] = []
    values["loop_time"] = []
    values["state_no_loop"] = []
