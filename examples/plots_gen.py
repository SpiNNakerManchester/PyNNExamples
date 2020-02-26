import glob
import matplotlib.pyplot as plt

values = {"dma_read": [],
          "state_update": [],
          "loop_time": [],
          "syn_events": [],
          "state_update_input": [],
          "loop_time_input": []}

for txtfile in glob.iglob("/localhome/g90604lp/ICPP_res/*.txt"):
    with open(txtfile, "r") as fp:
        i = 0
        partitions = []
        for line in fp:
            if i == 0:
                neurons = line.split(" ")[1][:-1]
                filename_split = txtfile.split("/")
                timestep = filename_split[len(filename_split)-1].split("_")[2]

            if i % 2 == 0:
                partitions.append(int(line.split(" ")[0]))
            else:
                dma_read = line.split(" ")[0]
                state_update = line.split(" ")[1]
                loop_time = line.split(" ")[2]
                syn_events = line.split(" ")[3]
                state_update_input = line.split(" ")[4]
                loop_time_input = line.split(" ")[5]
                values["dma_read"].append(float(dma_read))
                values["state_update"].append(float(state_update))
                values["loop_time"].append(float(loop_time))
                values["syn_events"].append(float(syn_events))
                values["state_update_input"].append(float(state_update_input))
                values["loop_time_input"].append(float(loop_time_input))
            i += 1

    plt.subplot(3, 2, 1)
    plt.plot(partitions, values["dma_read"], "o-")
    plt.title("DMA read timings")

    plt.subplot(3, 2, 2)
    plt.plot(partitions, values["syn_events"], "o-")
    plt.title("Synaptic events")

    plt.subplot(3, 2, 3)
    plt.plot(partitions, values["state_update"], "o-")
    plt.title("Neuron state update timings")

    plt.subplot(3, 2, 4)
    plt.plot(partitions, values["state_update_input"], "o-")
    plt.title("Input Neuron state update timings")

    plt.subplot(3, 2, 5)
    plt.plot(partitions, values["loop_time"], "o-")
    plt.title("Synaptic contribution sum timing")

    plt.subplot(3, 2, 6)
    plt.plot(partitions, values["loop_time_input"], "o-")
    plt.title("Input Synaptic contribution sum timing")
    plt.xlabel('number of synapse cores')

    plt.suptitle("Run with " + neurons + " Neurons, timestep " + timestep + " ms")
    plt.show()

    values["dma_read"] = []
    values["state_update"] = []
    values["loop_time"] = []
    values["syn_events"] = []
    values["state_update_input"] = []
    values["loop_time_input"] = []