import numpy as np
import matplotlib.pyplot as plt


def frozen_poisson_variable_hz(num_repeats, cycle_time, split, speed_up, pop_size):
    pattern = np.load("./1000 neurons - 10Hz - 100*1024s.npy", allow_pickle=True).tolist()

    if split > pop_size:
        print("Cannot split population into sizes smaller than 1 i.e. split > population size")
        raise Exception

    spikes = []
    l=[]
    for i in range(pop_size):
        l = []
        for j in range(len(pattern[i])):
            if pattern[i][j] < cycle_time:# * speed_up:
                l.append(float(pattern[i][j]) / speed_up)
            else:
                break
        spikes.append(l)

    speed_up_offset = cycle_time / split
    split_spikes = [[] for i in range(pop_size)]
    for neuron in range(len(spikes)):
        for spike in range(len(spikes[neuron])):
            split_spikes[neuron].append((spikes[neuron][spike] / split) + (speed_up_offset * int(neuron / (pop_size / split))))

    cycled_spikes = [[] for i in range(pop_size)]
    for r in range(0, num_repeats):
        for p in range(pop_size):
            new_iter = [i + r * cycle_time for i in split_spikes[p]]
            cycled_spikes[p].extend(new_iter)

    return cycled_spikes


cycle_time = 2200
repeats = 15
pop_size = 8
spike_times = frozen_poisson_variable_hz(repeats, cycle_time, 8, 10, pop_size)

scatterable_spike_times = [[], []]
for neuron in range(len(spike_times)):
    for spike_time in range(len(spike_times[neuron])):
        scatterable_spike_times[0].append(spike_times[neuron][spike_time])
        scatterable_spike_times[1].append(neuron)
        spike_times[neuron][spike_time] = int(spike_times[neuron][spike_time])

instantaneous_rate = 0
rate_decay_over_time = []
decay = np.exp(-1./10000.)
total_spikes = 0
running_average = []
reset_time = cycle_time
runtime = cycle_time*repeats
time_since_reset = 0

for time_step in range(1, runtime):
    if time_step % reset_time == 0:
        total_spikes = 0
        time_since_reset = 0
    time_since_reset += 1
    instantaneous_rate *= decay
    for neuron in range(len(spike_times)):
        for spike_time in spike_times[neuron]:
            if spike_time == time_step:
                total_spikes += 1
                instantaneous_rate += 1
            elif spike_time > time_step:
                break
    rate_decay_over_time.append(instantaneous_rate / pop_size)
    running_average.append((total_spikes / (time_since_reset / 1000.)) / pop_size)

fig, axs = plt.subplots(2, 1)
axs[0].set_title('rates calculated different ways')
axs[0].plot([i for i in range(len(rate_decay_over_time))], rate_decay_over_time, label='exponential')
axs[0].plot([i for i in range(len(running_average))], running_average, label='running average')
axs[0].legend(loc="bottom right")
axs[0].plot([0, runtime], [10, 10], 'r')
axs[1].scatter(scatterable_spike_times[0], scatterable_spike_times[1])
axs[0].set_xlim([-100, runtime+100])
axs[1].set_xlim([-100, runtime+100])
plt.show()
print("done")