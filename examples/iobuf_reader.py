import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

def profile_read_per_core(input_directory='./',profile_tags=['empty_rows_profile']):
    # profile_data_files = glob.glob(input_directory + '*.npz')
    times_dict = {}
    profile_data_files = sorted(glob.glob(input_directory + '*.npz'), key=lambda entry: int(str.split(str.split(entry, '/')[-1], '_')[0]))
    total_profile_times = []
    max_p = 0
    max_profile = None
    min_p = np.inf
    min_profile = None
    for i,filename in enumerate(profile_data_files):
        placement = str.split(filename, '/')[-1][:-12]
        xml_file = str.split(glob.glob(input_directory + placement + '*.xml')[0], '/')[-1][:-4]
        n_neurons = 1 + (int(str.split(xml_file, '_')[-1]) - int(str.split(xml_file, '_')[-2]))
        if n_neurons == 255:
        # if n_neurons == 255 and "octopus" not in xml_file:
            tag_times=[]
            profile_data = np.load(filename)
            for j,profile_tag in enumerate(profile_tags):
                measured_data = profile_data[profile_tag][:-1]
                if measured_data.max() > max_p:# and measured_data.min()>0:
                    max_profile = (xml_file+profile_tag,measured_data)
                    max_p = measured_data.max()
                if measured_data.min() < min_p:
                    min_profile = (xml_file+profile_tag,measured_data)
                    min_p = measured_data.min()
                tag_times.append(sum(measured_data))
            total_profile_times.append(tag_times)
            times_dict[xml_file]=sum(tag_times)
    return  total_profile_times,times_dict,max_profile,min_profile


def profile_read(input_directory='./',profile_tags=['empty_rows_profile']):
    total_profile_times = [[] for _ in range(len(profile_tags))]
    for filename in glob.glob(input_directory + '*.npz'):
        profile_data = np.load(filename)
        for j,profile_tag in enumerate(profile_tags):
            profile_times = profile_data[profile_tag]
            for t in profile_times:
                total_profile_times[j].append(t)
    return  total_profile_times


# input_directory = ''
def read_iobuf(input_directory='./'):
    empty_fractions = []
    empty_row_counts = []
    row_counts = []
    pop_counts = []
    spike_processing_counts = []
    empty_pop_counts = []
    for filename in glob.glob(input_directory+'*.txt'):
        f = open(filename)
        n_neurons = 0
        for line in f:
            if "n_neurons =" in line:
                n_neurons = int(float(str.split(line)[-1]))
                break
        if n_neurons==255:
            for line in f:
                if "empty fraction" in line:
                    empty_fractions.append(float(str.split(line)[-1]))
                if "pop count" in line and "empty pop count" not in line:
                    pop_counts.append(float(str.split(line)[-1]))
                if "empty row count" in line:
                    empty_row_counts.append(float(str.split(line)[-1]))
                if "nonzero row count" in line:
                    row_counts.append(float(str.split(line)[-1]))
                # if "invalid_master_pop_hits" in line:
                if "n_ghost_input_spikes" in line:
                    empty_pop_counts.append(float(str.split(line, '=')[-1]))
                if "spike_processing_count" in line:
                    spike_processing_counts.append(float(str.split(line, '=')[-1]))

    return empty_fractions,spike_processing_counts,row_counts,pop_counts,empty_pop_counts,empty_row_counts

def read_iobuf_bitfield(input_directory='./'):
    empty_row_counts = []
    bitfield_miss_counts = []
    empty_pop_counts = []
    for filename in glob.glob(input_directory+'*.txt'):
        f = open(filename)
        for line in f:
            if "empty row count" in line:
                empty_row_counts.append(float(str.split(line)[-1]))
            if "bitfield miss count" in line:
                bitfield_miss_counts.append(float(str.split(line)[-1]))
            if "invalid_master_pop_hits" in line:
                empty_pop_counts.append(float(str.split(line,'=')[-1]))

    return bitfield_miss_counts,empty_row_counts,empty_pop_counts

input_sizes = ['1k','2k','4k','8k','16k','32k']#,'30k']
# input_sizes = ['16k']
#no conn_lut
# input_directories = ['/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/conn_lut_disabled/2019-03-12-17-34-28-842578/run_1/provenance_data/',
#                     '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/conn_lut_disabled/2019-03-12-17-45-09-302854/run_1/provenance_data/' ]
#conn_lut
# input_directories = ['/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/conn_lut_enabled/2019-03-12-17-34-28-842578/run_1/provenance_data/',
#                      '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/conn_lut_enabled/2019-03-12-17-45-09-302854/run_1/provenance_data/']
#bitfield on
# input_directories = ['/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_on/2019-03-14-10-24-13-245906/run_1/provenance_data/',
#                      '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_on/2019-03-14-10-24-19-449456/run_1/provenance_data/',
#                      '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_on/2019-03-14-10-24-24-480353/run_1/provenance_data/',
#                      '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_on/2019-03-14-10-24-28-715968/run_1/provenance_data/',
#                      '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_on/2019-03-14-10-24-32-727758/run_1/provenance_data/']
#bitfield off
# input_directories = ['/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_off/2019-03-13-18-37-35-619445/run_1/provenance_data/',
#                      '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_off/2019-03-13-19-06-31-86460/run_1/provenance_data/',
#                      '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_off/2019-03-13-19-06-36-964813/run_1/provenance_data/',
#                      '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_off/2019-03-13-19-06-43-635597/run_1/provenance_data/',
#                      '/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/bitfield_off/2019-03-13-19-06-49-593681/run_1/provenance_data/']

#edge filtering on bitfield off
# input_directories = ['/home/rjames/SpiNNaker_devel/Brainstem/reports_local/reports/2019-03-14-13-32-09-391178/run_1/provenance_data/']
# input_directories = ['/home/rjames/SpiNNaker_devel/Brainstem/reports_local/reports/2019-03-14-14-36-52-72118/run_1/provenance_data/']


#edge filtering off distance dep
input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/distance_conn/edge_filter_off_bitfield_off/*'))]
# input_directories = input_directories[:-1]
#edge filtering on
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/distance_conn/edge_filter_on_bitfield_off/*'))]
# input_directories = input_directories[:-1]

# input_directories = ['/home/rjames/SpiNNaker_devel/PyNN8Examples/examples/reports_local/reports/2019-03-15-17-50-46-600670/run_1/provenance_data/']
# input_directories = ['/home/rjames/SpiNNaker_devel/PyNN8Examples/examples/reports_local/reports/2019-03-15-17-51-01-286377/run_1/provenance_data/']

# input_directories = ['/home/rjames/SpiNNaker_devel/PyNN8Examples/examples/reports_local/reports/2019-03-18-13-18-19-948776/run_1/provenance_data/']
# input_directories = ['/home/rjames/SpiNNaker_devel/PyNN8Examples/examples/reports_local/reports/2019-03-18-09-59-53-189879/run_1/provenance_data/',
#                      '/home/rjames/SpiNNaker_devel/PyNN8Examples/examples/reports_local/reports/2019-03-18-10-01-01-581093/run_1/provenance_data/']
#16k
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/distance_conn/16k compare/*'))]
# input_directories = input_directories[:-1]
#8k
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/distance_conn/8k compare/*'))]
#default fixed n_pre
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/fixed_n_100/default/*'))]

#bitfield + edge filter on
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/distance_conn/edge_filter_on_bitfield_on/*'))]

#Ear sims
#edge filter + bitfield
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/spinnak_ear/edge_filter_bitfield/*'))]
#default
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/spinnak_ear/default/*'))]
#edge filter
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/spinnak_ear/edge_filter/*'))]

#4k comparison
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/spinnak_ear/4k_cn/*'))]

#test
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/distance_conn/test/*'))]

#multiple rbn fixed prob
#default
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/fixed_prob/multiple_rbn/default/*'))]
#edge bitfield
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/fixed_prob/multiple_rbn/edge_filter_bitfield/*'))]
#test
input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/fixed_prob/multiple_rbn/test/*'))]

# plt.figure('bit field miss vs empty row counts')
# bf_miss,er,ep = read_iobuf_bitfield(input_directories[0])
# plt.plot(bf_miss)
# plt.plot(er)
# plt.xlabel("core ID")
# plt.ylabel("count")
# plt.legend(['bit field miss count','empty row count'])
# plt.show()

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:d} ({:.1f}%)".format(absolute,pct)


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
# plt.figure("incoming spike analysis")
input_labels = input_sizes
# input_labels = ['16k RBN all-to-all edges','filtered edges (~40%)']
fig, axs = plt.subplots(len(input_directories), 1)#,subplot_kw=dict(aspect="equal"))
if len(input_directories) == 1:
    axs = [axs]
for i,in_dir in enumerate(input_directories):
    res = read_iobuf(in_dir)
    index = len(input_directories)-i -1
    # plt.subplot(len(input_directories),1,len(input_directories)-i)
    # ax=plt.subplot(len(input_directories),1,i+1)
    # plt.title("$N_{pre}$ = " + input_sizes[i])
    # plt.pie([sum(res[1]),sum(res[2])],autopct='%1.1f%%')
    total_spike_count = sum(res[1])
    row_hit_count = sum(res[2])
    empty_pop_count = sum(res[4])
    empty_row_count = sum(res[5])
    axs[index].set_title(input_labels[i] + " total incoming spikes={}".format(int(total_spike_count)),fontsize=12)
    # axs.set_title(input_labels[i] + " total incoming spikes={}".format(int(total_spike_count)),fontsize=12)
    data = [total_spike_count-row_hit_count,row_hit_count]
    data = [empty_pop_count,empty_row_count,row_hit_count]
    labels = ["empty edge", "unconnected\nneuron",'']
    # labels = ["unconnected\nneuron",'']
    # axs.pie(data,autopct=lambda pct: func(pct, data))
    # axs[index].pie(data,autopct='%1.1f%%')
    # axs.pie(data,autopct='%1.1f%%')
    # plt.pie(data,autopct='%1.1f%%',colors=[colors[0],colors[0],colors[1]],explode=[0.1,0.1,0])
    # plt.pie(data,autopct=lambda pct: label_func(pct, labels),colors=[colors[0],colors[0],colors[1]],explode=[0.1,0.1,0])
    wedges,texts,pct_text=axs[index].pie(data,autopct='%1.1f%%',colors=[colors[0],colors[0],colors[1]],explode=[0.1,0.1,0],labels=labels,labeldistance=1.1,radius=0.9)
    # wedges,texts,pct_text=axs.pie(data,autopct='%1.1f%%',colors=[colors[0],colors[0],colors[1]],explode=[0.1,0.1,0],labels=labels,labeldistance=1.1,radius=0.9)
    # for text in texts:
    #     text.set_fontsize(10)



# # plt.subplot(len(input_directories),1,i+1)
# # plt.legend(["total empty row count","total nonzero row count"],bbox_to_anchor=(0.75, 0.1))
# plt.legend(["total unconnected spike count","total connected spike count"],bbox_to_anchor=(0.75, 0.1))
# legend_line=[plt.Line2D([0], [0], color=colors[0], lw=4),plt.Line2D([0], [0], color=colors[1], lw=4)]
# # plt.legend(["total empty edge spike count","total unconnected neuron spike count","total connected neuron spike count"],bbox_to_anchor=(0.75, 0.1))
# plt.legend(legend_line,["total unconnected spike count","total connected spike count"],bbox_to_anchor=(0.75, 0.1))


# average_incoming_spikes = []
# sd_incoming_spikes = []
# for i,in_dir in enumerate(input_directories):
#     res = read_iobuf(in_dir)
#     index = len(input_directories)-i -1
#     average_incoming_spikes.append(np.mean(res[1]))
#     sd_incoming_spikes.append(np.std(res[1]))
# input_directories = [x+'/run_1/provenance_data/' for x in sorted(glob.glob('/home/rjames/Dropbox (The University of Manchester)/EarProject/SpiNNaker_scale_tests/profile_results/distance_conn/edge_filter_on_bitfield_off/*'))]
# input_directories = input_directories[:-1]
# average_incoming_spikes_filter = []
# sd_incoming_spikes_filter = []
# for i,in_dir in enumerate(input_directories):
#     res = read_iobuf(in_dir)
#     average_incoming_spikes_filter.append(np.mean(res[1]))
#     sd_incoming_spikes_filter.append(np.std(res[1]))
#
# plt.figure()
# n = [1000*2**i for i in range(len(input_directories))]
# # plt.plot(n,average_incoming_spikes)
# plt.errorbar(n,average_incoming_spikes,yerr=sd_incoming_spikes,ls='dotted')
# plt.errorbar(n,average_incoming_spikes_filter,yerr=sd_incoming_spikes_filter,ls='dotted')
# plt.title("incoming spikes with RBN scale increase")
# plt.ylabel("average number of incoming spikes per core")
# plt.xlabel("size of input population")
# # plt.ylim((0,300e3))
# plt.grid()
# plt.legend(['all-to-all edges','filtered edges (~40%)'],loc=2)
# plt.show()

t_stellate_max_profiles = [[] for _ in range(len(input_directories))]
d_stellate_max_profiles = [[] for _ in range(len(input_directories))]
octopus_max_profiles = [[] for _ in range(len(input_directories))]
bushy_max_profiles = [[] for _ in range(len(input_directories))]
max_times = [[] for _ in range(len(input_directories))]
min_times = [[] for _ in range(len(input_directories))]
max_profile_legend=[]
min_profile_legend=[]
for i,in_dir in enumerate(input_directories[:]):
    profile_tags = ['spike_profile','timer_profile']
    profile_times,times_dict,max_profile,min_profile = profile_read_per_core(in_dir,profile_tags=profile_tags)
    if max_profile is not None:
        plt.figure("max profile")
        print "max profile {}:  {}".format(input_sizes[i],max_profile[0])
        plt.plot(max_profile[1])
        max_profile_legend.append(input_sizes[i])
    if 0:#min_profile is not None:
        plt.figure("min profile")
        print "min profile {}:  {}".format(input_sizes[i],min_profile[0])
        plt.plot(min_profile[1])
        min_profile_legend.append(input_sizes[i])

    plt.figure("profiles")
    av_times=np.mean(profile_times,axis=0)
    sd=np.std(profile_times,axis=0)
    width = 0.7/len(input_directories)
    # ind = np.arange(len(profile_tags))
    ind = np.arange(1)
    for j,av in enumerate(av_times):
        if j == 0:
            spike_pro = plt.bar(ind + (i * width) - 0.28, av_times[j], width=width, yerr=sd[j],
                                    align='center',color=colors[i],hatch='x',edgecolor='black')[0]

        else:
            timer_pro=plt.bar(ind + (i * width) - 0.28, av_times[j], width=width, yerr=sd[j], align='center',
                              bottom=spike_pro._height,color=colors[i],edgecolor='black')[0]
            ax = plt.gca()
            ax.text(spike_pro._x0 + (spike_pro._width / 4), 1.02 * (spike_pro._height+timer_pro._height), input_sizes[i],
                    horizontalalignment='left',
                    verticalalignment='center', color='black', weight='bold')




    # plt.ylim((0,2000))
    # plt.ylim((0,200))
    # print av_times
    # print sd
    for key in times_dict:
        if "t_stellate" in key:
            t_stellate_max_profiles[i].append(times_dict[key])
        if "d_stellate" in key:
            d_stellate_max_profiles[i].append(times_dict[key])
        if "octopus" in key:
            octopus_max_profiles[i].append(times_dict[key])
        if "bushy" in key:
            bushy_max_profiles[i].append(times_dict[key])

    max_times[i].append(max(times_dict.values()))
    min_times[i].append(min(times_dict.values()))

plt.ylabel('average total execution time per core (ms)')
# plt.title('SpiNNaker simulation with fast connectivity look-up')
plt.title('SpiNNaker simulation with DMA based look-up')
# plt.title('16k SpiNNak-Ear CN simulation')
# plt.title('RBN SpiNNaker simulation fixed n synapses = 100 \n'
#           'distance dependent connectivity with edge filtering')
# plt.title('RBN SpiNNaker simulation fixed n synapses = 100 \n'
#           'distance dependent connectivity with edge filtering \n'
#           '+ fast connectivity look-up')
# plt.title('RBN SpiNNaker simulation fixed n synapses = 100 \n'
#           'distance dependent connectivity')
# plt.title('16k RBN SpiNNaker fixed n synapses = 100 \n'
#           'distance dependent connectivity')
# profile_tags_labels = ['spike processing','synapse + neuron update']
profile_tags_labels = ['spike processing + synapse/neuron update']
plt.xticks(ind, profile_tags_labels)
# plt.yticks(np.arange(0, 81, 10))
legend_labels = input_sizes[:len(input_directories)]
# legend_labels = ['default','with optimisations']
# plt.legend(ps, legend_labels[:len(input_directories)],loc=2)
# plt.legend((ps[0][0], ps[1][0], ps[2][0]), ['all-to-all edges','filtered edges (~40%)','filtered edges (~40%)\n + fast connectivity look-up'],loc=2)

plt.figure("max profile")
plt.title('max profiles for SpiNNaker simulation with DMA based look-up')
plt.ylabel("time to complete timer callback (ms)")
plt.legend(max_profile_legend)
# plt.legend(legend_labels)
# plt.show()
#     plt.subplot(len(input_sizes),1,i+1)
#     plt.title("$N_{pre}$ = " + input_sizes[i])
#     plt.hist(profile_times)
# plt.legend(["total empty row count","total nonzero row count"],bbox_to_anchor=(0.75, 1.))

plt.show()
