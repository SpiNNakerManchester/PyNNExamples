import numpy as np
import matplotlib.pyplot as plt
# add comment

n = 1000
x = np.zeros(n)
t = np.arange(n)

# Setup signal as combination of sin waves of different frequency
input_freq_1 = 2
input_freq_2 = 25
amp_1 = 1
amp_2 = 3.75

for i in t:
    x[i] = amp_1 * np.sin(input_freq_1 * i * (2 * np.pi / n)) \
         + amp_2 * np.sin(input_freq_2 * i * (2 * np.pi / n))


FT = np.fft.fft(x)/(n/2) # Compute and normalise FT
FT = FT[range(n/2)] # FT is symmetric so only need half

# Extract frequencies to print
freqs = np.fft.fftfreq(n)

# print abs(FT)
print 'FT frequencies'
for f in range(len(FT)):
    if (abs(FT[f]) > 0.001):
        print freqs[f]*n

# Plot signal and FT
fig, ax = plt.subplots(2,1)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[0].plot(t,x)
ax[1].plot(t[range(n/2)],abs(FT[range(n/2)])) # plot half as symmetric
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('|FT|')
plt.subplots_adjust(hspace=0.4)
plt.show()






