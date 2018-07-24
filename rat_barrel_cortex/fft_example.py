import numpy as np
import matplotlib.pyplot as plt

n = 1000
x = np.zeros(n)
t = np.arange(n)

# Setup signal as combination of sin waves of different frequency
input_freq_1 = 2
input_freq_2 = 30

for i in t:
    x[i] = np.sin(input_freq_1 * i * (2 * np.pi / n)) \
         + np.sin(input_freq_2 * i * (2 * np.pi / n))

# Compute and normalise FT
FT = np.fft.fft(x)/n

# Plot signal and FT
fig, ax = plt.subplots(2,1)
ax[0].plot(t,x)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(t[range(n/2)],abs(FT[range(n/2)])) # plot half as symmetric
ax[0].set_xlabel('Frequency')
ax[0].set_ylabel('|FT|')
plt.show()





