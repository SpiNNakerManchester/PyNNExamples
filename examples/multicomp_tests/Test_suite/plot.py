import matplotlib.pyplot as plt


with open("/localhome/g90604lp/out.txt", "r") as fp:

    values = {}
    j = 0
    while j < 5:
        fp.readline()
        name = fp.readline()
        fp.readline()
        k = 0
        elems = {}
        while k < 7:
            line = fp.readline()
            l = line.split(" ")
            postsyn = int(l[0].split("-")[1])
            time = float(l[1]) / 200
            elems[postsyn] = time
            k += 1
        values[name] = elems
        j += 1

for key in values.keys():
    print(sorted(values[key].values()))
    plt.plot(sorted(values[key].keys()), sorted(values[key].values()), "-o", linewidth=2)
    plt.xlabel("Target neurons")
    plt.ylabel("Time (micros)")
    plt.grid(True)
    plt.title(key)
    plt.show()