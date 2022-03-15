import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

with open("/localhome/g90604lp/out_mnist.txt", "r") as fp:
    file = fp.read().split(" uS\n")
    stripped = []
    for i in range(len(file) - 1):
        rd = file[i].strip("[] \n").split(" ")
        try:
            line = []
            for n in rd:
                if n != "":
                    line.append(float(n))
        except:
            print(rd)
            print(file[i])
            print(str(i))
            print("\n\n\n\n")
        stripped.append(line)

plotting = np.transpose(stripped)

x = [_ for _ in range(len(stripped))]

data = MNIST('/localhome/g90604lp/datasets')

test_img, test_lab = data.load_testing()

filtered_test_lab = []

for i in range(len(test_lab)):
    if test_lab[i] == 0 or test_lab[i] == 1 or test_lab[i] == 2:
        filtered_test_lab.append(test_lab[i])

print(filtered_test_lab)

for i in range(len(plotting)):
    plt.plot(x[len(stripped) - 31470 : len(stripped)], plotting[i][len(stripped) - 31470 : len(stripped)], label="n " + str(i))

plt.legend()
plt.grid(True)
plt.show()