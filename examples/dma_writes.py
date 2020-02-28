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
            values["times"].append(float(line.split(" ")[1]))


    plt.plot(values["cores"], values["times"], "o-")
    plt.title("DMA write timings")

    plt.show()
