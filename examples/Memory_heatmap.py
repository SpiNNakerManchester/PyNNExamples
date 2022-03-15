import matplotlib.pyplot as plt
import numpy as np

datasetmax = [[0 for i in range(1, 16)] for i in range(1, 16)]
datasetmin = [[0 for i in range(1, 16)] for i in range(1, 16)]
datasetavg = [[0 for i in range(1, 16)] for i in range(1, 16)]
font_size = 19.5

filename = "/localhome/g90604lp/memory_times_SDRAM_fin.txt"

filelist = filename.split("/")
filetype = filelist[3].split("_")

with open(filename, "r") as fp:
    line = fp.readline()
    line = fp.readline()
    while line != "":
        l = line.split(" ")
        n_core = int(l[0].strip("(,"))
        s_core = int(l[1].strip(")"))
        time = float(l[2].strip("[]\n"))
        datasetmax[n_core][s_core] = round(time, 2)
        if filetype[0] == "syn":
            datasetmin[n_core][s_core] = round(float(l[3].strip("[]\n")), 2)
            datasetavg[n_core][s_core] = round(float(l[4].strip("[]\n")), 2)
        else:
            datasetmin[n_core][s_core] = round(float(l[4].strip("[]\n")), 2)
            datasetavg[n_core][s_core] = round(float(l[6].strip("[]\n")), 2)
        line = fp.readline()

fig, ax = plt.subplots()
im = ax.imshow(datasetmax, vmin=0, vmax=27)

cbar = ax.figure.colorbar(im, ax=ax)

if filetype[0] == "syn":
    cbar.ax.set_ylabel("Memory Write Times ($\mu$s)", rotation=-90, va="bottom", fontsize=font_size-2)

    cbar.ax.tick_params(labelsize=font_size)
    plt.rcParams.update({'font.size': font_size-4.5})
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax.set_xlabel("Synapse cores", fontsize=font_size)
    ax.set_ylabel("Neuron cores", fontsize=font_size)

    ax.set_xlabel("Synapse cores")
    ax.set_ylabel("Neuron cores")

    if filetype[2] == "SysRAM":
        mem = "SysRAM and SDRAM"
    else:
        mem = "SDRAM"
    ax.set_title(mem + " Writing times (max)", fontsize=font_size)

else:
    cbar.ax.set_ylabel("Memory Read Times ($\mu$s)", rotation=-90, va="bottom", fontsize=font_size-2)

    cbar.ax.tick_params(labelsize=font_size)
    plt.rcParams.update({'font.size': font_size-4.5})
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax.set_xlabel("Synapse cores", fontsize=font_size)
    ax.set_ylabel("Neuron cores", fontsize=font_size)

    ax.set_xlabel("Synapse cores")
    ax.set_ylabel("Neuron cores")

    if filetype[2] == "SysRAM":
        mem = "SysRAM and SDRAM"
    else:
        mem = "SDRAM"
    ax.set_title(mem + " Reading Times (max)", fontsize=font_size)

# Loop over data dimensions and create text annotations.
for i in range(len(datasetmax)):
    for j in range(len(datasetmax[i])):
        if datasetmax[i][j] != 0:
            text = ax.text(j, i, datasetmax[i][j],
                           ha="center", va="center", color="w", rotation=-45)

plt.show()

if filetype[0] == "syn":

    fig, ax = plt.subplots()
    im = ax.imshow(datasetmin, vmin=0, vmax=27)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Memory Write Times ($\mu$s)", rotation=-90, va="bottom", fontsize=font_size-2)

    cbar.ax.tick_params(labelsize=font_size)
    plt.rcParams.update({'font.size': font_size-4.5})
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax.set_xlabel("Synapse cores", fontsize=font_size)
    ax.set_ylabel("Neuron cores", fontsize=font_size)

    ax.set_xlabel("Synapse cores")
    ax.set_ylabel("Neuron cores")

    if filetype[2] == "SysRAM":
        mem = "SysRAM and SDRAM"
    else:
        mem = "SDRAM"
    ax.set_title(mem + " Writing times (min)", fontsize=font_size)

    # Loop over data dimensions and create text annotations.
    for i in range(len(datasetmin)):
        for j in range(len(datasetmin[i])):
            if datasetmin[i][j] != 0:
                text = ax.text(j, i, datasetmin[i][j],
                               ha="center", va="center", color="w", rotation=-45)

    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(datasetavg, vmin=0, vmax=27)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Memory Write Times ($\mu$s)", rotation=-90, va="bottom", fontsize=font_size-2)

    cbar.ax.tick_params(labelsize=font_size)
    plt.rcParams.update({'font.size': font_size-4.5})
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax.set_xlabel("Synapse cores", fontsize=font_size)
    ax.set_ylabel("Neuron cores", fontsize=font_size)

    ax.set_xlabel("Synapse cores")
    ax.set_ylabel("Neuron cores")

    if filetype[2] == "SysRAM":
        mem = "SysRAM and SDRAM"
    else:
        mem = "SDRAM"
    ax.set_title(mem + " Writing times (avg)", fontsize=font_size)

    # Loop over data dimensions and create text annotations.
    for i in range(len(datasetavg)):
        for j in range(len(datasetavg[i])):
            if datasetavg[i][j] != 0:
                text = ax.text(j, i, datasetavg[i][j],
                               ha="center", va="center", color="w", rotation=-45)

    plt.show()
else:

    fig, ax = plt.subplots()
    im = ax.imshow(datasetmin, vmin=0, vmax=27)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Memory Read Times ($\mu$s)", rotation=-90, va="bottom", fontsize=font_size-2)

    cbar.ax.tick_params(labelsize=font_size)
    plt.rcParams.update({'font.size': font_size-4.5})
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax.set_xlabel("Synapse cores", fontsize=font_size)
    ax.set_ylabel("Neuron cores", fontsize=font_size)

    if filetype[2] == "SysRAM":
        mem = "SysRAM and SDRAM"
    else:
        mem = "SDRAM"
    ax.set_title(mem + " Reading times (min)", fontsize=font_size)

    # Loop over data dimensions and create text annotations.
    for i in range(len(datasetmin)):
        for j in range(len(datasetmin[i])):
            if datasetmin[i][j] != 0:
                text = ax.text(j, i, datasetmin[i][j],
                               ha="center", va="center", color="w", rotation=-45)

    plt.show()

    fig, ax = plt.subplots()
    im = ax.imshow(datasetavg, vmin=0, vmax=27)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Memory Read Times ($\mu$s)", rotation=-90, va="bottom", fontsize=font_size-2)

    cbar.ax.tick_params(labelsize=font_size)
    plt.rcParams.update({'font.size': font_size-4.5})
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    ax.set_xlabel("Synapse cores", fontsize=font_size)
    ax.set_ylabel("Neuron cores", fontsize=font_size)

    ax.set_xlabel("Synapse cores")
    ax.set_ylabel("Neuron cores")

    if filetype[2] == "SysRAM":
        mem = "SysRAM and SDRAM"
    else:
        mem = "SDRAM"
    ax.set_title(mem + " Reading times (avg)", fontsize=font_size)

    # Loop over data dimensions and create text annotations.
    for i in range(len(datasetavg)):
        for j in range(len(datasetavg[i])):
            if datasetavg[i][j] != 0:
                text = ax.text(j, i, datasetavg[i][j],
                               ha="center", va="center", color="w", rotation=-45)

    plt.show()

