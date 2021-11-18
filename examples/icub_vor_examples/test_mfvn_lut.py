from spynnaker.pyNN.models.neuron.plasticity.stdp.common import write_mfvn_lut

beta = 10
sigma = 200

write_mfvn_lut(spec=None,
               sigma=sigma,
               beta=beta,
               lut_size=256,
               shift=0,
               time_probe=22)
