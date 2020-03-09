import glob
import matplotlib.pyplot as plt

values = {"syn_events": []}

for txtfile in glob.iglob("/localhome/g90604lp/ICPP_res/synapses/*.txt"):
    with open(txtfile, "r") as fp:
        i = 0
        partitions = []
        for line in fp:
            if i == 0:
                #neurons = line.split(" ")[1][:-1]
                filename_split = txtfile.split("/")
                timestep = filename_split[len(filename_split)-1].split("_")[2]

            if i % 2 == 0:
                partitions.append(int(line.split(" ")[0]))
            else:
                syn_events = line.split(" ")[0]
                values["syn_events"].append(float(syn_events))

            i += 1
        neurons = txtfile.split("/")[5].split("_")[0]


    plt.plot(partitions, values["syn_events"], "o-")
    plt.title("Synaptic events")

    plt.xlabel('number of synapse cores')

    plt.suptitle("Run with " + neurons + " Neurons, timestep " + timestep + " ms")
    plt.show()

    values["syn_events"] = []