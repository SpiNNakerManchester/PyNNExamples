"""
retina example that just feeds data from a retina to live output via an
intermediate population
"""
try:
    import pyNN.spiNNaker as p
except Exception as e:
    import spynnaker8 as p

connected_chip_details = {
    "spinnaker_link_id": 0,
}


def get_updated_params(params):
    params.update(connected_chip_details)
    return params


# Setup
p.setup(timestep=1.0)

# FPGA Retina - Down Polarity
retina_pop = p.Population(
    None, p.external_devices.ExternalFPGARetinaDevice, get_updated_params({
        'retina_key': 0x5,
        'mode': p.external_devices.ExternalFPGARetinaDevice.MODE_128,
        'polarity': (
            p.external_devices.ExternalFPGARetinaDevice.DOWN_POLARITY)}),
    label='External retina')

population = p.Population(256, p.IF_curr_exp(), label='pop_1')
p.Projection(
    retina_pop, population, p.FixedProbabilityConnector(0.1),
    synapse_type=p.StaticSynapse(weight=0.1))

# q.activate_live_output_for(population)
p.run(1000)
p.end()
