import glob
import matplotlib.pyplot as plt

values = {"cores": [],
          "times": []}

for txtfile in glob.iglob("/localhome/g90604lp/dma_writes/*.txt"):
    with open(txtfile, "r") as fp:
        i = 0
        cores = []
        times = []
        for line in fp:
            values["cores"].append(float(line.split(" ")[0]))
            values["times"].append(float(line.split(" ")[1]) / 1000)


    plt.plot(values["cores"], values["times"], "o-")
    plt.grid()
    plt.xlabel("cores", size=28)
    plt.ylabel("Time (microseconds)", size=28)
    plt.xticks(size=17)
    plt.yticks(size=17)

    plt.show()
