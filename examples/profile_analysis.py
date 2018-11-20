import numpy as np
import matplotlib.pyplot as plt

output_file_prefix = '1k_optimised_'
mp = np.load('./master_profiles.npz')
n_calls = mp['n_calls']

profile_s_index = 0
spike_profiles = []
profile_d_index = 0
dma_profiles = []
for i,n_call in enumerate(n_calls):
    if i%2 == 0:
        to_index = int(profile_s_index+n_call)
        spike_profiles.append(mp['spike_processing'][profile_s_index:to_index])
        profile_s_index=to_index
    else:
        to_index = int(profile_d_index+n_call)
        dma_profiles.append(mp['dma_processing'][profile_d_index:to_index])
        profile_d_index=to_index




total_dma_time = [np.sum(core) for core in dma_profiles]
total_spike_time = [np.sum(core) for core in spike_profiles]


np.savez_compressed('./'+output_file_prefix+'dma_spike_processing_profile.npz',total_dma_time=total_dma_time,total_spike_time=total_spike_time,dma_profiles=dma_profiles,spike_profiles=spike_profiles)

#=======================================================================
#---------GRAPH PLOT----------------------------------------------------
#=======================================================================
def plot_profile(n_neurons,test_duration,legend_dict):
    optimised = np.load('./'+n_neurons+'_optimised_dma_spike_processing_profile.npz')
    default = np.load('./'+n_neurons+'_default_dma_spike_processing_profile.npz')
    # edge = np.load('./'+n_neurons+'_edge_dma_spike_processing_profile.npz')

    total_dma_time=[optimised['total_dma_time'],default['total_dma_time']]#,edge['total_dma_time']]
    total_spike_time=[optimised['total_spike_time'],default['total_spike_time']]#,edge['total_spike_time']]
    plt.figure("DMA profile")
    for times in total_dma_time:
        x = np.linspace(0,1,len(times))
        plt.plot(x,100*(times/test_duration))
        plt.ylabel('percentage of total simulation time')
        plt.xticks(np.arange(2), ('0', 'n'))
        plt.xlabel('machine vertices')
        plt.title('Total DMA read time')
    legend_dict['DMA'].append("optimised"+n_neurons)
    legend_dict['DMA'].append("default"+n_neurons)
    # legend_dict['DMA'].append("edge"+n_neurons)
    plt.legend(legend_dict['DMA'])
    plt.figure("spike processing profile")
    for times in total_spike_time:
        x = np.linspace(0,1,len(times))
        plt.plot(x,100*(times/test_duration))
        plt.ylabel('percentage of total simulation time')
        plt.xlabel('machine vertices')
        plt.xticks(np.arange(2), ('0', 'n'))
        plt.title('Total spike processing time')

    legend_dict['SPIKES'].append("optimised"+n_neurons)
    legend_dict['SPIKES'].append("default"+n_neurons)
    # legend_dict['SPIKES'].append("edge"+n_neurons)
    plt.legend(legend_dict['SPIKES'])

tests = ['1k','10k']
test_duration = 1000.

legend_dict={}
legend_dict['DMA']=[]
legend_dict['SPIKES']=[]

for test in tests:
    plot_profile(test,test_duration,legend_dict)

plt.show()
