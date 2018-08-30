import numpy
import math
from scipy.io import wavfile
from scipy.signal import resample
from matplotlib.ticker import FormatStrFormatter

def generate_signal(signal_type="tone",fs=22050.,dBSPL=40.,
                    freq=3000.,duration=0.5,ramp_duration=0.003,
                    silence_duration=0.05,modulation_freq=0.,
                    modulation_depth=1.,plt=None,file_name=None, silence=True,title='',ascending=True):
    T = 1./fs
    amp = 1. * 28e-6 * 10. ** (dBSPL / 20.)
    num_samples = numpy.ceil(fs * duration)
    # inverted amp on input gives rarefaction effect for positive pressure (?!))
    if signal_type == "tone":
        map_bs_cos_shift = numpy.pi/2
        signal = [-amp*(numpy.sin(2*numpy.pi*freq*T*i+map_bs_cos_shift) *
                  (modulation_depth*0.5*(1+numpy.cos(2*numpy.pi*modulation_freq*T*i))))#modulation
                  for i in range(int(num_samples))]
        #map_bs_remove 1st sample!?
        signal=signal[1:]

    elif signal_type == "sweep_tone":
        if len(freq)<2:
            print "enter low and high frequency sweep values as freq=[low,high]"
        phi = 0
        f = freq[0]
        delta = 2. * numpy.pi * f * T
        # f_delta = (freq[1] - freq[0]) /num_samples
        f_delta = numpy.power((freq[1] / freq[0]),1./num_samples)# 1.0002534220677803#(freq[1] / freq[0]) /num_samples
        signal = []
        for i in range(int(num_samples)):
            signal.append(-amp * numpy.sin(phi))
            phi = phi + delta
            f = f * f_delta
            # f = f + f_delta
            delta = 2. * numpy.pi * f * T
	if not ascending:
	    signal = signal[::-1]
    elif signal_type == "file":
        #silence = False
        if file_name:
            [fs_f,signal] = wavfile.read(file_name)
            if len(signal.shape)>1:#stereo
                signal = signal[:, 0]
            fs_f=numpy.float32(fs_f)
            if fs_f != fs:
                secs = len(signal)/fs_f
                num_resamples = secs * fs
                signal = resample(signal,num_resamples)

            signal = numpy.float32(signal)
            max_val=numpy.max(numpy.abs(signal))
            for i in range(len(signal)):
                # if signal[i] == max_val:
                #     print
                signal[i]/=max_val
                signal[i]*=-amp #set loudness
        else:
            raise Exception("must include valid wav filename")

    else:
        print "invalid signal type!"
        signal = []

    # add ramps
    num_ramp_samples = numpy.round(fs * ramp_duration)
    #step = (numpy.pi / 2.0) / (num_ramp_samples - 1)
    step = numpy.pi / (num_ramp_samples - 1)

    for i in range(int(num_ramp_samples)):
        #ramp = (numpy.sin(i * step))
        ramp = (1 + numpy.cos(numpy.pi + i * step))/2.
        # on ramp
        signal[i] *= ramp
        # off ramp
        signal[-i] *= ramp
    if silence:
        # add silence
        num_silence_samples = int(numpy.ceil(fs*silence_duration))
        signal = numpy.concatenate((numpy.zeros(num_silence_samples),signal,numpy.zeros(num_silence_samples)))

    if plt:
        plt.figure(title)
        time = numpy.linspace(0,len(signal)/fs,len(signal))
        plt.plot(time,signal)
        plt.xlim((0,len(signal)/fs))
        #plt.plot(signal)
        plt.xlabel("time (s)")
        plt.ylabel("signal amplitude")
    return signal

def generate_psth_8(target_neuron_ids,spike_trains,bin_width,
                  duration,scale_factor=0.001):
    num_bins = numpy.ceil(duration/bin_width)
    psth = numpy.zeros([len(target_neuron_ids),num_bins])
    psth_row_index=0
    for i in target_neuron_ids:
        #extract target neuron times and scale
        spike_times = spike_trains[i]
        scaled_times= [spike_time.item() * scale_factor for spike_time in spike_times if spike_time*scale_factor<=duration]
        scaled_times.sort()
        bins = numpy.arange(bin_width,duration,bin_width)
        for j in scaled_times:
            idx = (numpy.abs(bins - j)).argmin()
            if bins[idx] < j:
                idx+=1
            psth[psth_row_index][idx] += 1

        #increment psth_row_index
        psth_row_index += 1

    sum= numpy.sum(psth,axis=0)
    mean = sum/psth_row_index
    output = [count * 1./bin_width for count in mean]
    return output

def generate_psth(target_neuron_ids,spike_trains,bin_width,
                  duration,scale_factor=0.001):
    num_bins = numpy.ceil(duration/bin_width)
    psth = numpy.zeros([len(target_neuron_ids),num_bins])
    psth_row_index=0
    for i in target_neuron_ids:
        #extract target neuron times and scale
        spike_times = [spike_time for (neuron_id, spike_time) in spike_trains if neuron_id==i]
        scaled_times= [spike_time * scale_factor for spike_time in spike_times if spike_time*scale_factor<=duration]
        scaled_times.sort()
        bins = numpy.arange(bin_width,duration,bin_width)
        for j in scaled_times:
            idx = (numpy.abs(bins - j)).argmin()
            if bins[idx] < j:
                idx+=1
            psth[psth_row_index][idx] += 1

        #increment psth_row_index
        psth_row_index += 1

    sum= numpy.sum(psth,axis=0)
    mean = sum/psth_row_index
    output = [count * 1./bin_width for count in mean]
    return output

def psth_plot(plt,target_neuron_ids,spike_trains,bin_width,
                  duration,scale_factor=0.001,title='PSTH'):
    PSTH = generate_psth(target_neuron_ids, spike_trains, bin_width=bin_width,
                            duration=duration, scale_factor=scale_factor)
    x = numpy.linspace(0, duration, len(PSTH))
    plt.figure(title)
    plt.plot(x,PSTH)
    plt.ylabel("firing rate (sp/s)")
    plt.xlabel("time (s)")

def psth_plot_8(plt, target_neuron_ids, spike_trains, bin_width,
              duration, scale_factor=0.001, title='PSTH',filepath=None):
    PSTH = generate_psth_8(target_neuron_ids, spike_trains, bin_width=bin_width,
                         duration=duration, scale_factor=scale_factor)
    x = numpy.linspace(0, duration, len(PSTH))
    plt.figure(title)
    plt.plot(x, PSTH)
    max_rate = max(PSTH)
    plt.ylim((0,max_rate+1))
    plt.ylabel("firing rate (sp/s)")
    plt.xlabel("time (s)")
    if filepath is not None:
        plt.savefig(filepath + '/{}.eps'.format(title))

def spike_raster_plot(spikes,plt,duration,ylim,scale_factor=0.001,title=None):
    if len(spikes) > 0:
        spike_times = [spike_time for (neuron_id, spike_time) in spikes]
        scaled_times = [spike_time * scale_factor for spike_time in spike_times]
        spike_ids = [neuron_id for (neuron_id, spike_time) in spikes]
        spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]

        ##plot results
        plt.figure(title)
        plt.plot(scaled_times, spike_ids, '.', markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0)
        plt.ylim(0, ylim)
        plt.xlim(0, duration)
        plt.ylabel("neuron ID")
        plt.xlabel("time (s)")

def spike_raster_plot_8(spikes,plt,duration,ylim,scale_factor=0.001,title=None,filepath=None,xlim=None,
                        onset_times=None,pattern_duration=None):
    if len(spikes) > 0:
        neuron_index = 1
        spike_ids = []
        spike_times = []

        for times in spikes:
            for time in times:
                spike_ids.append(neuron_index)
                spike_times.append(time)
            neuron_index+=1
        scaled_times = [spike_time * scale_factor for spike_time in spike_times]

        ##plot results
        plt.figure(title)
        plt.plot(scaled_times, spike_ids, '.', markersize=3,
                 markerfacecolor='black', markeredgecolor='none',
                 markeredgewidth=0)
        plt.ylim(0, ylim)
        plt.xlim(0, duration)
        plt.ylabel("neuron ID")
        plt.xlabel("time (s)")

        if onset_times is not None:
            #plot block of translucent colour per pattern
            ax = plt.gca()
            pattern_legend=[]
            legend_labels=[]
            colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k','w']
            # labels = ['A','B','C',]
            for i,pattern in enumerate(onset_times):
                pattern_legend.append(plt.Line2D([0], [0], color=colours[i%8], lw=4,alpha=0.2))
                legend_labels.append("s{}".format(i+1))
                for onset in pattern:
                    x_block = (onset,onset+scale_factor*pattern_duration)
                    ax.fill_between(x_block,ylim,alpha=0.2,facecolor=colours[i%8],lw=0.5)
            plt.legend(pattern_legend,legend_labels,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                        ncol=len(onset_times), mode="expand", borderaxespad=0.)
        if xlim is not None:
            plt.xlim(xlim)
        if filepath is not None:
            plt.savefig(filepath + '/{}.pdf'.format(title))#switched to pdf as using transparent images


def multi_spike_raster_plot(spikes_list,plt,duration,ylim,scale_factor=0.001,marker_size=3,dopamine_spikes=[],title=''):
    plt.figure(title)
    marker_colours = ['black','blue','red','green']
    marker_styles = ['.','+','*','o']
    count = 0
    for spikes in spikes_list:
        if len(spikes) > 0:
            spike_times = [spike_time for (neuron_id, spike_time) in spikes]
            scaled_times = [spike_time * scale_factor for spike_time in spike_times]
            spike_ids = [neuron_id for (neuron_id, spike_time) in spikes]
            spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]

            ##plot results
            plt.plot(scaled_times, spike_ids, marker_styles[count], markersize=marker_size,
                     color=marker_colours[count])
            plt.ylim(0, ylim)
            plt.xlim(0, duration)
        count +=1
    if len(dopamine_spikes) > 0:
            spike_times = [spike_time for (neuron_id, spike_time) in dopamine_spikes]
            scaled_times = [spike_time * scale_factor for spike_time in spike_times]
            for xc in scaled_times:
                plt.axvline(x=xc,color='red')
            plt.ylim(0, ylim)
            plt.xlim(0, duration)

def weight_array_to_group_list(weight_array,from_ids,to_ids):
    group_weights_to = []
    group_weights_from = []
    for id in from_ids:  # group_target_ids:
        connection_weights_from = [weight for weight in weight_array[id][:] if not math.isnan(weight)]
        group_weights_from.append(numpy.array(connection_weights_from))
    for id in to_ids:
        connection_weights_to = [weight for weight in weight_array[:][id] if not math.isnan(weight)]
        group_weights_to.append(numpy.array(connection_weights_to))
    return [group_weights_to,group_weights_from]

# def vary_weight_plot(varying_weights,ids,stim_ids,duration,plt,num_recs,np,ylim,title=''):
#     if len(ids)>0:
#         #varying_weights_array = np.array(varying_weights)
#         repeats = np.linspace(0, duration, num_recs)
#         sr = math.sqrt(len(ids))
#         num_cols = np.ceil(sr)
#         num_rows = np.ceil(len(ids)/num_cols)
#
#         plt.figure(title)
#         plt.suptitle("weight updates for all connections")
#
#         count=0
#         for id in ids:
#             plt.subplot(num_rows,num_cols,count+1)
#             id_times = []
#             for reading in varying_weights:
#                 id_times.append(reading[id])
#
#             id_times = numpy.array(id_times)
#             if len(id_times.shape)>1:
#                 id_times=map(list, zip(*id_times))
#                 for t in id_times:
#                     plt.plot(repeats,t)
#             else:
#                 plt.plot(repeats,id_times)
#             #weights=varying_weights_array[:,id]
#             #every number of connections per neuron over time should be equal(no struc plasticity)
#             #insane way to get each time element from the weights list
#             # if len(weights.shape)>1:
#             #     times = np.zeros((len(weights[0]),len(weights)))
#             #     for i in range(len(weights[0])):
#             #         for j in range(len(weights)):
#             #             times[i,j]=weights[j][i]
#             #     for t in times:
#             #         plt.plot(repeats, t)
#             # else:
#             #     plt.plot(repeats, weights)
#
#             label = plt.ylabel("ID:{}".format(str(id+1)))
#             if id in stim_ids:
#                 label.set_color('red')
#             plt.xlim(0,duration)
#             plt.ylim(0,ylim)
#             count+=1

def vary_weight_plot(varying_weights,ids,stim_ids,duration,plt,num_recs,np,ylim,title='',filepath=None):
    if len(varying_weights)!=num_recs:
        raise Exception("incorrect number of weight recordings taken (num_recs={}, len(varyingweights={})".format(num_recs,len(varying_weights)))
    if len(ids)>0:
        repeats = np.linspace(0, duration, num_recs)
        sr = math.sqrt(len(ids))
        num_cols = np.ceil(sr)
        num_rows = np.ceil(len(ids)/num_cols)
        if num_rows >1 and num_cols > 1:
            figsize = (5*num_cols,3*num_rows)
        else:
            figsize = (8,6)

        if stim_ids:
            plt.figure(title,figsize=figsize)
            plt.suptitle("non-pattern synapse weight updates for post neurons")
            plt.figure(title + "pattern",figsize=figsize)
            plt.suptitle("pattern synapse weight updates for post neurons")
        else:
            plt.figure(title,figsize=figsize)
            plt.suptitle("synapse weight updates for post neurons")

        count=0
        for id in ids:
            if stim_ids:
                id_times_pattern = [[] for _ in range(num_recs)]
            else:
                id_times_pattern = []
            id_times = [[] for _ in range(num_recs)]
            rec_index = 0
            for reading in varying_weights:
                for (pre, post, weight) in reading:
                    if post == id:
                        if stim_ids and pre in stim_ids:
                            id_times_pattern[rec_index].append(weight)
                        else:
                            id_times[rec_index].append(weight)
                rec_index +=1


            plt.figure(title)
            plt.subplot(num_rows,num_cols,count+1)
            id_times = numpy.array(id_times)
            if len(id_times.shape)>1:
                id_times=map(list, zip(*id_times))
                for t in id_times:
                    plt.plot(repeats,t)
                label = plt.ylabel("ID:{}".format(str(id+1)))
                plt.xticks(np.linspace(0,duration,5))
                if num_rows > 2:
                    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    plt.yticks(np.linspace(0,ylim,3))
                else:
                    plt.xlabel('time (s)')
                plt.xlim(0,duration)
                plt.ylim(0,ylim)
            if id_times_pattern:
                plt.figure(title + "pattern")
                plt.subplot(num_rows, num_cols, count + 1)
                id_times_pattern = numpy.array(id_times_pattern)
                if len(id_times_pattern.shape) > 1:
                    id_times_pattern = map(list, zip(*id_times_pattern))
                    for t in id_times_pattern:
                        plt.plot(repeats, t)
                label = plt.ylabel("ID:{}".format(str(id + 1)))
                plt.xticks(np.linspace(0,duration,5))
                if num_rows > 2:
                    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    plt.yticks(np.linspace(0,ylim,3))
                else:
                    plt.xlabel('time (s)')
                plt.xlim(0, duration)
                plt.ylim(0, ylim)
            #else:
            #    plt.plot(repeats,id_times)
            count+=1
        if filepath is not None:
            plt.figure(title)
            plt.savefig(filepath+'/{}_weights.eps'.format(title))
            if stim_ids:
                plt.figure(title + "pattern")
                plt.savefig(filepath+'/{}_pattern_weights.eps'.format(title))

def weight_dist_plot(varying_weights,num_ticks,plt,w_min,w_max,np=numpy,title=None,filepath=None):
    #varying_weights_array = np.array(varying_weights)
    #initial_weights = varying_weights_array[0][:]
    # init_weights = []
    # if len(initial_weights.shape)>1:
    #     for weights in initial_weights:
    #         for weight in weights:
    #             init_weights.append(weight)
    #     init_weights = np.asarray(init_weights)
    #     final_weights = varying_weights_array[-1][:]
    #     fin_weights = []
    #     for weights in final_weights:
    #         for weight in weights:
    #             fin_weights.append(weight)
    # else:
    #     init_weights = varying_weights_array[0][:]
    #     fin_weights = varying_weights_array[-1][:]
    #assumes varying weights is a list of connectionsholders, for a distribution plot we don't care about the ids
    v_weights = []
    for neuron,reading in enumerate(varying_weights):
        v_weights.append([])
        for (pre, post, weight) in reading:
            v_weights[neuron].append(weight)

    initial_weights = numpy.array(v_weights[0])
    if len(initial_weights.shape) > 1:
        init_weights = []
        for weights in initial_weights:
            for weight in weights:
                init_weights.append(weight)
        final_weights = v_weights[-1]
        fin_weights = []
        for weights in final_weights:
            for weight in weights:
                fin_weights.append(weight)
    else:
        init_weights = v_weights[0]
        fin_weights = v_weights[-1]
    plt.figure(title)
    plt.hist(init_weights,bins=100,alpha=0.5,range=(w_min,w_max))
    plt.hist(fin_weights,bins=100,alpha=0.5,range=(w_min,w_max))
    plt.legend(["first recording", "last recording"])
    plt.ylabel("number of synapses")
    plt.xlabel("weight of synapse")
    if filepath is not None:
        plt.savefig(filepath + '/'+title+'.pdf')#switched to pdf as using transparent images


def cell_voltage_plot_8(v, plt, duration_ms, time_step_ms,scale_factor=0.001, id=None, title='',filepath=None):
    # times = range(0,int(duration_ms),int(time_step_ms))
    membrane_voltage = v[0]
    times = range(0,membrane_voltage.shape[0])
    scaled_times = [time*scale_factor for time in times]
    if id is not None:
        mem_v = [v_t[id] for v_t in membrane_voltage]
        title = title + str(id + 1)
    else:
        mem_v = membrane_voltage
        title = title + "{} neurons".format(membrane_voltage.shape[1])
    plt.figure(title)
    plt.plot(scaled_times, mem_v)
    plt.xlim((0,duration_ms*scale_factor))
    plt.xlabel('time (s)')
    plt.ylabel('membrane voltage (mV)')
    if filepath is not None:
        plt.savefig(filepath + '/' + title + '_memV.eps')

def cell_voltage_plot(v,plt,duration,scale_factor=0.001,id=0,title=''):
        times = [i[1] for i in v if i[0]==id]
        scaled_times = [time * scale_factor for time in times]
        membrane_voltage = [i[2] for i in v if i[0]==id]
        plt.figure(title + str(id+1))
        plt.plot(scaled_times,membrane_voltage)

#function to create a normal probability distribution of distance connectivity list
def distance_dependent_connectivity(pop_size,weights,delays,min_increment=0,max_increment=1):
    pre_index = 0
    conns = []
    while pre_index < pop_size:
        #
        increment = numpy.unique(
            numpy.round(abs(numpy.random.normal(loc=0, scale=numpy.sqrt(max_increment - 1), size=max_increment))))
        limited_increment = [inc for inc in increment if inc >= min_increment and inc < max_increment]#increment[(increment >= min_increment) & (increment < max_increment)]
        for inc in limited_increment:
            post_index = pre_index + inc
            if post_index < pop_size:
                conns.append((pre_index, post_index, weights[int(inc)],delays[int(inc)]))

        rev_increment = numpy.unique(
            numpy.round(abs(numpy.random.normal(loc=0, scale=numpy.sqrt(max_increment - 1), size=max_increment))))
        limited_rev_increment = [inc for inc in rev_increment if inc >= min_increment and inc < max_increment]#rev_increment[(rev_increment >= min_increment) & (rev_increment < max_increment)]
        for r_inc in limited_rev_increment:
            rev_post_index = pre_index - r_inc
            if rev_post_index >= 0:
                conns.append((pre_index, rev_post_index, weights[int(r_inc)],delays[int(r_inc)]))

        pre_index += 1

    return conns

def fixed_prob_connector(num_pre,num_post,p_connect,weights,delays=1.,self_connections=False,single_pop=False):
    connection_list=[]
    #do pre->post connections
    num_connections_per_neuron = int(num_post*p_connect)
    for id in range(num_pre):
        choice = range(num_post)
        if single_pop:
            #exclude current id from being chosen
            choice.remove(id)
        connections = numpy.random.choice(choice,num_connections_per_neuron,replace=False)
        for target_id in connections:
            connection_list.append((id,target_id,weights,delays))

    if not single_pop:
        # do post->pre connections
        num_connections_per_neuron = num_pre * p_connect
        for id in range(num_post):
            choice = range(num_pre)
            if single_pop:
                # exclude current id from being chosen
                choice.remove(id)
            connections = numpy.random.choice(choice, num_connections_per_neuron, replace=False)
            for target_id in connections:
                connection_list.append((id, target_id, weights, delays))

    return connection_list

def test_filter(audio_data,b0,b1,b2,a0,a1,a2):
    past_input=numpy.zeros(2)
    past_concha=numpy.zeros(2)
    concha=numpy.zeros(len(audio_data))
    for i in range(441,len(audio_data)):
        # if i>=1202:
            # print ''
        concha[i]=(b0 * audio_data[i]
                  + b1 * audio_data[i-1]#past_input[0]
                  + b2 * audio_data[i-2]#past_input[1]
                  - a1 * concha[i-1]#past_concha[0]
                  - a2 * concha[i-2]#past_concha[1]
                     ) * a0

        #past_input[1] = past_input[0]
        #past_input[0] = audio_data[i]

        #past_concha[1] = past_concha[0]
        #past_concha[0] = concha[i]
    return concha

#connects two spike trains from the same AN model in time
def spike_train_join(spike_trains,num_neurons):
    #spike trains is a list of pynn compatable spike source array spike trains
    #e.g. [[[.,.,.],[.,.,.]],[[.,.,.],[.,.,.]]]

    #output is a single pynn compatable spike source array spike train
    #e.g. [[.,.,.],[.,.,.]]
    spike_train_output=[]
    for i in range(num_neurons):
        spike_train_output.append([])

    max_time = 0
    for spike_train in spike_trains:
        if len(spike_train)!=num_neurons:
            raise Exception("number of neurons mismatch")
        index = 0
        new_max = 0
        for neuron in spike_train:
            for spike_time in neuron:
                later_time = spike_time+max_time
                spike_train_output[index].append(later_time)
                if later_time > new_max:
                    new_max = later_time
            index+=1
        #record final time to start from for next spike train
        #max_time = numpy.amax(spike_train_output)
        #max_time = max_time[-1]
        max_time=new_max

    return [spike_train_output,max_time]

def normal_dist_connection_builder(pre_size,post_size,RandomDistribution,
                                   rng,conn_num,dist,sigma,conn_weight,delay=1.):
    conn_list = []

    for post in xrange(post_size):
        #mu = int(dist / 2) + post * dist
        mu = dist / 2. + post * dist
        an2ch = RandomDistribution('normal', (mu, sigma), rng=rng)
        an_idxs = an2ch.next(n=conn_num)
        pre_check = []
        for pre in an_idxs:
            pre = int(pre)
            if pre >= 0 and pre < pre_size:
                if pre not in pre_check:
                    if type(conn_weight)==float:
                        weight = conn_weight
                    else:#assumes rand dist
                        weight = conn_weight.next(n=1)
                    if type(delay)!=float:
                        conn_delay = delay.next(n=1)
                    else:
                        conn_delay = delay
                    conn_list.append((pre, int(post), weight, conn_delay))
                pre_check.append(pre)

    return conn_list
#find stimulus onset times from audio signal
def audio_stimulus_onset_detector(audio_signal,Fs,num_classes):
    import numpy as np
    #average samples over previous 10ms absolute values
    num_10ms_samples = Fs*0.01
    envelope_upper_threshold = 0.00005
    envelope_lower_threshold = 0.00002
    stimulus_times = []
    for j in range(num_classes):
        stimulus_times.append([])
    class_index=0
    triggered = False
    for i,sample in enumerate(audio_signal):
        if i>=num_10ms_samples:
            envelope = np.mean(np.absolute(audio_signal[int(i-num_10ms_samples):i]))
            if envelope > envelope_upper_threshold and triggered == False:
                stimulus_times[class_index].append(i/Fs)
                if class_index < (num_classes - 1):
                    class_index += 1
                else:
                    class_index = 0
                triggered = True
            if envelope < envelope_lower_threshold and triggered == True:
                triggered = False

    return  stimulus_times
#find stimulus onset times from AN spikes
#assumes interleaved class presentations (a,b,c,a,b,c)
def stimulus_onset_detector(spike_train_an_ms,num_an_fibres,duration,num_classes):
    #calculate psth with 10ms bin widths across all AN fibres to get average full spectrum response
    PSTH = generate_psth_8(range(num_an_fibres), spike_train_an_ms, bin_width=0.01,
                         duration=duration, scale_factor=0.001)

    stimulus_times = []
    for j in range(num_classes):
        stimulus_times.append([])
    class_index = 0
    time_index = 0
    x = numpy.arange(0, duration, duration / float(len(PSTH)))
    triggered = False
    for rate in PSTH:
        if rate > 50. and triggered == False:
            stimulus_times[class_index].append(x[time_index] * 1000)
            if class_index < (num_classes - 1):
                class_index += 1
            else:
                class_index = 0
            triggered = True
        if rate < 30 and triggered == True:
            triggered = False
        time_index += 1

    return stimulus_times

# stimulus onset times is a list of onset times lists for each stimulus
# spike time is the output spikes from a population
def neuron_correlation(spike_train,time_window, stimulus_onset_times,max_id,noise_threshold,np=numpy,significant_spike_count=None):
    correlations = []#1st dimension is stimulus class
    # #counts = np.asarray([np.zeros(max_id + 1), np.zeros(max_id + 1)])
    counts=[]
    # stimulus_index = 0
    # for stimulus in stimulus_onset_times:
    #     correlations.append([])
    #     counts.append(np.zeros(max_id + 1))
    #     #loop through each stimulus onset time and check all neuron firing times
    #     #save neuron index if firing is within time window of stimulus onset
    #     # for time in stimulus:
    #     #     for (neuron_id, spike_time) in spike_train:
    #     #         if spike_time > time and spike_time <= (time + time_window) and (neuron_id,time) not in correlations[stimulus_index]:
    #     #             correlations[stimulus_index].append((neuron_id,time))
    #     for time in stimulus:
    #         for neuron_id,neuron in enumerate(spike_train):
    #             for spike_time in neuron:
    #                 if spike_time > time and spike_time <= (time + time_window) and (neuron_id+1,spike_time) not in correlations[stimulus_index]:
    #                     correlations[stimulus_index].append((neuron_id+1,spike_time))
    for stimulus_index, stimulus in enumerate(stimulus_onset_times):
        correlations.append([[] for _ in range(int(max_id+1))])
        counts.append(np.zeros(max_id + 1))
        # counts.append(np.zeros(max_id + 1))
        # loop through each stimulus onset time and check all neuron firing times
        # save neuron index if firing is within time window of stimulus onset
        for t_index, time in enumerate(stimulus):
            for neuron in correlations[stimulus_index]:
                neuron.append(0)
            for (neuron_id, spike_time) in spike_train:
                if spike_time > time and spike_time <= (time + time_window):
                    correlations[stimulus_index][int(neuron_id)][t_index] += 1

        #check which neurons have fired above or below the noise threshold
        for id,neuron in enumerate(np.asarray(correlations[stimulus_index])):
            #generate count of neuron firings that are above the noise threshold
            counts[stimulus_index][id] = np.count_nonzero(neuron > noise_threshold)

    if significant_spike_count is None:
        significant_spike_count = np.mean(counts,axis=1)  # 6.#

    selective_neuron_ids = selective_id_finder(counts, significant_spike_count)

    return np.asarray(counts), selective_neuron_ids, significant_spike_count  # correlations

def selective_id_finder(counts, significant_spike_count):
    selective_neuron_ids = [[] for _ in range(len(counts))]
    for i, stimulus in enumerate(counts):
        for id_count, count in enumerate(stimulus):
            if count >= significant_spike_count[i]:
                others = range(len(counts))
                others.remove(i)
                # check neuron doesn't respond to other stimuli
                # ensures neuron response is exclusive to a single class
                exclusive = True
                for j in others:
                    if counts[j][id_count] > 0:
                        exclusive = False
                if exclusive:
                    selective_neuron_ids[i].append(id_count)
            id_count += 1

    return selective_neuron_ids

def selective_neuron_search(pattern_spikes,spike_train,time_window,final_pattern_start,
                            plt,filepath=None,np=numpy,significant_spike_count=None):

    #take final 10% of pattern spikes
    stimulus_times=[]
    for pattern in pattern_spikes:
        stimulus_times.append([time for time in pattern if time>=final_pattern_start])
    final_spike_train=[]
    for train in spike_train:
        final_spike_train.append([time for time in train if time>=final_pattern_start])

    max_id = len(spike_train)
    counts,selective_neuron_ids,significant_spike_count = neuron_correlation(final_spike_train,time_window,
                                                                             stimulus_times,max_id,significant_spike_count=significant_spike_count)

    print "significant spike count: {}".format(significant_spike_count)
    max_count = counts.max()
    plt.figure(figsize=(20,10))
    title = "{}ms post-stimulus spike count for target layer".format(time_window)
    plt.title(title)
    plt.xlabel("neuron ID")
    plt.ylabel("spike count")
    plt.plot(counts.T)
    legend_string=[]
    for i in range(len(stimulus_times)):
        legend_string.append("stimulus {}".format(i+1))
    plt.legend(legend_string)
    plt.ylim((0,max_count+1))

    for i in range(len(selective_neuron_ids)):
        print selective_neuron_ids[i]

    if filepath is not None:
        plt.savefig(filepath+"/{}.eps".format(title))
        import csv
        from itertools import izip_longest
        with open(filepath+"/selective_neurons.csv","w+") as f:
            writer = csv.writer(f)
            for values in izip_longest(*selective_neuron_ids):
                writer.writerow(values)

#assumes input connectivity in varying_weights format
# def connection_plot(varying_weights)

def connection_hist_plot(varying_weights,pre_size,post_size,plt,title='',filepath=None,weight_min=0.000001):
    import numpy as np
    incoming_connections=[[]for _ in range(post_size)]
    source_list=[]
    target_list=[]
    #take final reading
    final_connections = varying_weights[-1]
    for (source,target,weight) in final_connections:
        if source is not None and weight>weight_min:
            if source in incoming_connections[target]:
                print "multapse detected!"
            incoming_connections[target].append(source)
            source_list.append(source)
            target_list.append(target)
    out_figure = title+'pre_pop outgoing connections'
    plt.figure(out_figure)
    plt.hist(source_list,bins=pre_size,alpha=0.5,range=(0,pre_size))
    if filepath is not None:
        plt.savefig(filepath + '/'+out_figure+'.eps')
    in_figure = title+'post_pop incoming connections'
    plt.figure(in_figure)
    plt.hist(target_list,bins=post_size,alpha=0.5,range=(0,post_size))
    if filepath is not None:
        plt.savefig(filepath + '/'+in_figure+'.eps')

def sparsity_measure(onset_times,output_spikes,onset_window=5.,from_time=0):
    import numpy as np
    n_neurons = float(len(output_spikes))
    sparsity_matrix = [[] for _ in range(len(onset_times))]
    # np_output_spikes = [[0.]for _ in range(int(n_neurons))]
    # for id, neuron in enumerate(output_spikes):
    #     for spike in neuron:
    #         np_output_spikes[id].append(spike.item())
    # np_output_spikes = np.asarray(np_output_spikes)

    #go through each stimulus onset time and bin all the subsequent output spike IDs that fall in onset time + onset window
    for id,stimulus in enumerate(onset_times):
        for time in stimulus:
            if time >= from_time:
                counts = np.zeros((int(n_neurons),int(onset_window)))
                for out_id,neuron in enumerate(output_spikes):
                    for output_spike in neuron:
                        # only care if at least one spike per neuron has occured in window
                        if output_spike >= time and output_spike < (time+onset_window): #and counts[out_id]==0.:
                            if isinstance(output_spike, (int,float)):
                                counts[out_id,int(output_spike-time-1)]+=1
                            else:#assume quantity
                                counts[out_id,int(output_spike.item()-time-1)]+=1
                #calculate sum of active neurons across presentation
                # sum =  np.sum(np.sum(counts,axis=0))
                sum =  np.sum(counts,axis=1)
                #average across timesteps in window
                av = np.mean(sum)
                # av = np.sum(sum)
                #record the percentage of total active IDs in each bin relative to the total number of neurons in output spikes
                # sparsity_matrix[id].append((sum/(n_neurons*onset_window))*100.)
                # sparsity_matrix[id].append((av/(n_neurons))*100.)
                sparsity_matrix[id].append(av)
    return sparsity_matrix