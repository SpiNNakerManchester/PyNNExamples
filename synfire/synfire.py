"""
Synfire chain example
"""
try:
    import pyNN.spiNNaker as p
except Exception as e:
    import spynnaker7.pyNN as p


# number of neurons in each population
n_neurons = 100
n_populations = 10
weights = 0.5
delays = 17.0

p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)

spikeArray = {'spike_times': [[0]]}
stimulus = p.Population(1, p.SpikeSourceArray, spikeArray, label='stimulus')

chain_pops = [
    p.Population(n_neurons, p.IF_curr_exp, {}, label='chain_{}'.format(i))
    for i in range(n_populations)
]
for pop in chain_pops:
    pop.record()

connector = p.FixedNumberPreConnector(10, weights, delays)
for i in range(n_populations):
    p.Projection(chain_pops[i], chain_pops[(i + 1) % n_populations], connector)

p.Projection(stimulus, chain_pops[0], p.AllToAllConnector(weights=5.0))

p.run(1000)
spikes = [pop.getSpikes() for pop in chain_pops]
p.end()

if __name__ == '__main__':
    try:
        import pylab
        pylab.figure()
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron')
        pylab.title('Spikes Sent By Chain')
        offset = 0
        for pop_spikes in spikes:
            pylab.plot(
                [i[1] for i in pop_spikes],
                [i[0] + offset for i in pop_spikes], "."
            )
            offset += n_neurons
        pylab.show()
    except Exception as ex:
        print ex
        print spikes
