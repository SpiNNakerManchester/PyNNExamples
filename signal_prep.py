import numpy
import math
from scipy.io import wavfile
# import scipy.signal
from nnresample import resample
from matplotlib.ticker import FormatStrFormatter

def generate_signal(signal_type="tone",fs=22050.,dBSPL=40.,
                    freq=3000.,duration=0.5,ramp_duration=0.003,
                    silence_duration=0.05,modulation_freq=0.,
                    modulation_depth=1.,plt=None,file_name=None, silence=True,title='',ascending=True,channel=0):
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
                signal = signal[:, channel]
            fs_f=numpy.float32(fs_f)
            if fs_f != fs:
                secs = len(signal)/fs_f
                num_resamples = secs * fs
                # signal = scipy.signal.resample(signal,int(num_resamples))
                # signal = scipy.signal.resample_poly(signal,int(num_resamples),len(signal))
                signal = resample(signal,fs,fs_f)

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
        #silence is realistic noise -20dBSPL
        #silence_amp = 1. * 28e-6 * 10. ** ((dBSPL - 60.) / 20.)
        silence_amp = 1. * 28e-6 * 10. ** (-20. / 20.)
        silence_samples = ((2*numpy.random.rand(num_silence_samples))-1.)*silence_amp
        signal_silence_samples = ((2*numpy.random.rand(len(signal)))-1.)*silence_amp
        signal = numpy.concatenate((silence_samples,signal+signal_silence_samples,silence_samples))
        # signal = numpy.concatenate((numpy.zeros(num_silence_samples),signal,numpy.zeros(num_silence_samples)))
    else:
        signal=numpy.asarray(signal)
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
                  duration):
    import numpy as np
    bins = np.arange(bin_width*1000., duration*1000., bin_width*1000.)
    if isinstance(spike_trains,list):
        spike_trains = np.asarray(spike_trains)
    target_neurons = spike_trains[target_neuron_ids]
    hist=[]
    for spike_times in target_neurons:
        hist.append(np.histogram(spike_times,bins=bins)[0])
    # psth = numpy.zeros([len(target_neuron_ids),num_bins])
    # psth_row_index=0
    # for i in target_neuron_ids:
    #     #extract target neuron times and scale
    #     spike_times = spike_trains[i]
    #     scaled_times= [spike_time.item() * scale_factor for spike_time in spike_times if spike_time*scale_factor<=duration]
    #     scaled_times.sort()
    #     bins = numpy.arange(bin_width,duration,bin_width)
    #     for j in scaled_times:
    #         idx = (numpy.abs(bins - j)).argmin()
    #         if bins[idx] < j:
    #             idx+=1
    #         psth[psth_row_index][idx] += 1
    #
    #     #increment psth_row_index
    #     psth_row_index += 1
    #
    # sum= numpy.sum(psth,axis=0)
    # mean = sum/psth_row_index
    # output = [count * 1./bin_width for count in mean]
    output = np.mean(hist,axis=0) * 1./bin_width
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
              duration,title='PSTH',filepath=None):
    PSTH = generate_psth_8(target_neuron_ids, spike_trains, bin_width=bin_width,
                         duration=duration)
    x = numpy.linspace(0, duration, len(PSTH))
    plt.figure(title)
    plt.plot(x, PSTH)
    max_rate = max(PSTH)
    plt.ylim((0,max_rate+1))
    plt.ylabel("firing rate (sp/s)")
    plt.xlabel("time (s)")
    if filepath is not None:
        plt.savefig(filepath + '/{}.eps'.format(title))
    return PSTH
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

def vary_weight_plot(varying_weights,ids,stim_ids,duration,plt,num_recs,np,ylim,title='',filepath=None,legend=None,figsize=None):
    if len(varying_weights)!=num_recs:
        raise Exception("incorrect number of weight recordings taken (num_recs={}, len(varyingweights={})".format(num_recs,len(varying_weights)))
    if len(ids)>0:
        repeats = np.linspace(0, duration, num_recs)
        sr = math.sqrt(len(ids))
        num_cols = np.ceil(sr)
        num_rows = np.ceil(len(ids)/num_cols)
        if num_rows >1 and num_cols > 1 and figsize is None:
            figsize = (5*num_cols,3*num_rows)
        elif figsize is None:
            figsize = (8,6)
        else:
            figsize = figsize
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
            if legend is not None:
                legend_string=[]
                legend_line=[]
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
                        if legend is not None and pre not in legend_string:
                            legend_string.append("pre_id:{}".format(pre+1))


                rec_index +=1


            plt.figure(title)
            plt.subplot(num_rows,num_cols,count+1)
            id_times = numpy.array(id_times)
            if len(id_times.shape)>1:
                id_times=map(list, zip(*id_times))
                for t in id_times:
                    line, = plt.plot(repeats,t)
                    if legend is not None:
                        legend_line.append(plt.Line2D([0], [0], color=line.get_color(), lw=4))
                if legend is not None:
                    plt.legend(legend_line, legend_string,
                               bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=3, mode="expand", borderaxespad=0.)
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
    plt.hist(init_weights,bins=100,alpha=0.5,range=(w_min,w_max*1.1))
    plt.hist(fin_weights,bins=100,alpha=0.5,range=(w_min,w_max*1.1))
    plt.legend(["first recording", "last recording"])
    plt.ylabel("number of synapses")
    plt.xlabel("weight of synapse")
    if filepath is not None:
        plt.savefig(filepath + '/'+title+'.pdf')#switched to pdf as using transparent images


def cell_voltage_plot_8(v, plt, duration_ms, time_step_ms,scale_factor=0.001, id=None, title='',filepath=None,subplots=None):
    # times = range(0,int(duration_ms),int(time_step_ms))
    membrane_voltage = v[0]
    times = range(0,membrane_voltage.shape[0])
    scaled_times = [time*scale_factor for time in times]
    if id is not None:
        mem_v = [v_t[id] for v_t in membrane_voltage]
        if isinstance(id,float):
            title = title + str(id + 1)
    else:
        mem_v = membrane_voltage
        title = title + "{} neurons".format(membrane_voltage.shape[1])
    if subplots is None:
        plt.figure(title)
    else:
        ax=plt.subplot(subplots[0],subplots[1],subplots[2])
        ax.set_title(title)
    plt.plot(scaled_times, mem_v)
    plt.xlim((0,duration_ms*0.001))
    if subplots is None:
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
def distance_dependent_connectivity(pop_size,weights=1.,delays=1.,min_increment=0,max_increment=1):
    post_index = 0
    conns = []
    # while pre_index < pop_size:
    #     #
    #     increment = numpy.unique(
    #         numpy.round(abs(numpy.random.normal(loc=0, scale=numpy.sqrt(max_increment - 1), size=max_increment))))
    #     limited_increment = [inc for inc in increment if inc >= min_increment and inc < max_increment]#increment[(increment >= min_increment) & (increment < max_increment)]
    #     for inc in limited_increment:
    #         post_index = pre_index + inc
    #         if post_index < pop_size:
    #             conns.append((pre_index, post_index, weights[int(inc)],delays[int(inc)]))
    #
    #     rev_increment = numpy.unique(
    #         numpy.round(abs(numpy.random.normal(loc=0, scale=numpy.sqrt(max_increment - 1), size=max_increment))))
    #     limited_rev_increment = [inc for inc in rev_increment if inc >= min_increment and inc < max_increment]#rev_increment[(rev_increment >= min_increment) & (rev_increment < max_increment)]
    #     for r_inc in limited_rev_increment:
    #         rev_post_index = pre_index - r_inc
    #         if rev_post_index >= 0:
    #             conns.append((pre_index, rev_post_index, weights[int(r_inc)],delays[int(r_inc)]))
    #
    #     pre_index += 1
    while post_index < pop_size:
        #
        increment = numpy.unique(
            numpy.round(abs(numpy.random.normal(loc=0, scale=numpy.sqrt(max_increment - 1), size=max_increment))))
        limited_increment = [inc for inc in increment if inc >= min_increment and inc < max_increment]#increment[(increment >= min_increment) & (increment < max_increment)]
        for inc in limited_increment:
            pre_index = post_index + inc
            if pre_index < pop_size:
                if isinstance(weights,float):
                    weight = weights
                else:
                    weight = weights.next(n=1)
                if isinstance(delays,float):
                    delay = delays
                else:
                    delay = delays.next(n=1)
                # conns.append((pre_index, post_index, weight,delay))
                conns.append((pre_index, post_index))

        # rev_increment = numpy.unique(
        #     numpy.round(abs(numpy.random.normal(loc=0, scale=numpy.sqrt(max_increment - 1), size=max_increment))))
        # limited_rev_increment = [inc for inc in rev_increment if inc >= min_increment and inc < max_increment]#rev_increment[(rev_increment >= min_increment) & (rev_increment < max_increment)]
        # for r_inc in limited_rev_increment:
        #     rev_post_index = pre_index - r_inc
        #     if rev_post_index >= 0:
        #         conns.append((pre_index, rev_post_index, weights[int(r_inc)],delays[int(r_inc)]))

        post_index += 1

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

#assumes ID is actually a position in 2D space i.e. the max post and pre IDs are the same regardless of how many neurons are in each pop
def spatial_normal_dist_connection_builder(spatial_range,n_pre,n_post,RandomDistribution,
                                   conn_num,dist,sigma,conn_weight=None,delay=1.,p_connect=1.0,
                                   delay_scale=None,posts=None,multapses=True):
    import numpy as np
    if posts is None:
        posts = xrange(n_post)

    conn_list = []
    post_scale = float(spatial_range)/n_post
    pre_scale = float(spatial_range)/n_pre

    for post in posts:
        scaled_post = int(post*post_scale)
        mu = int(dist / 2) + scaled_post * dist
        # mu = dist / 2. + post * dist
        pre_dist = RandomDistribution('normal_clipped',[mu,sigma,0,n_pre*pre_scale -1])
        if isinstance(conn_num,float) or isinstance(conn_num,int):
            number_of_connections = conn_num
        else:
            number_of_connections = conn_num.next(n=1)
        pre_idxs = pre_dist.next(n=int(number_of_connections))
        pre_check = []
        for pre in pre_idxs:
            scaled_pre = int(pre)#int(np.round(pre / pre_scale))
            # if scaled_pre >= 0 and scaled_pre < pre_size:
            if scaled_pre not in pre_check and np.random.rand() <= p_connect:
                if conn_weight is None:
                    conn_list.append((scaled_pre, scaled_post))
                else:
                    if type(conn_weight) == float:
                        weight = conn_weight
                    else:  # assumes rand dist
                        weight = conn_weight.next(n=1)
                    if type(delay) != float:
                        if delay_scale is not None:
                            conn_delay = int(delay.next(n=1)) * delay_scale
                        else:
                            conn_delay = delay.next(n=1)
                    else:
                        conn_delay = delay

                    conn_list.append((scaled_pre, scaled_post, weight, conn_delay))
            if multapses is False:
                pre_check.append(scaled_pre)

    return conn_list

def normal_dist_connection_builder(pre_size,post_size,RandomDistribution,
                                   conn_num,dist,sigma,conn_weight=None,delay=1.,p_connect=1.0,
                                   delay_scale=None,dist_weight=None,posts=None,multapses=True):
    import numpy as np
    if posts is None:
        posts = xrange(post_size)

    conn_list = []
    if pre_size > post_size:
        post_scale = float(pre_size)/post_size
        pre_scale = 1.
    else:
        post_scale = 1.
        pre_scale = float(post_size)/pre_size

    for post in posts:
        scaled_post = int(post*post_scale)
        mu = int(dist / 2) + scaled_post * dist
        # mu = dist / 2. + post * dist
        pre_dist = RandomDistribution('normal_clipped',[mu,sigma,0,pre_size*pre_scale -1])
        if isinstance(conn_num,float) or isinstance(conn_num,int):
            number_of_connections = conn_num
        else:
            number_of_connections = conn_num.next(n=1)
        pre_idxs = pre_dist.next(n=int(number_of_connections))
        pre_check = []
        for pre in pre_idxs:
            scaled_pre = int(np.round(pre / pre_scale))
            # if scaled_pre >= 0 and scaled_pre < pre_size:
            if scaled_pre not in pre_check and np.random.rand() <= p_connect:
                if conn_weight is None:
                    conn_list.append((scaled_pre, post))
                else:
                    if type(conn_weight) == float:
                        weight = conn_weight
                    elif dist_weight is not None:
                        # p_dist = np.exp(float(abs(scaled_pre - post))/(post_size*post_scale) -1.)
                        p_dist = float(abs(scaled_pre - post))/(post_size*post_scale)
                        weight = dist_weight * p_dist
                    else:  # assumes rand dist
                        weight = conn_weight.next(n=1)
                    if type(delay) != float:
                        if delay_scale is not None:
                            conn_delay = int(delay.next(n=1)) * delay_scale
                        else:
                            conn_delay = delay.next(n=1)
                    else:
                        conn_delay = delay

                    conn_list.append((scaled_pre, post, weight, conn_delay))
            if multapses is False:
                pre_check.append(scaled_pre)

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
            if source in incoming_connections[int(target)]:
                print "multapse detected!"
            incoming_connections[int(target)].append(source)
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

def connection_surface_plot(varying_weights,pre_size,post_size,plt,title='',filepath=None,n_plots=2):
    import numpy as np
    incoming_connections=[[]for _ in range(post_size)]

    plot_indices = [int(idx) for idx in np.linspace(0,len(varying_weights)-1,n_plots)]
    for i in plot_indices:
        final_connections = varying_weights[i]
        surface = np.zeros((pre_size,post_size))
        for (source,target,weight) in final_connections:
            if source in incoming_connections[int(target)]:
                print "multapse detected!"
            incoming_connections[int(target)].append(source)
            surface[source][target] += weight

        figure = title + ' connection weights {}'.format(i)
        plt.figure(figure)
        plt.imshow(surface,vmin=0,vmax=surface.max(),interpolation='none',origin='lower',cmap='viridis')
        plt.xlabel('target neuron')
        plt.ylabel('source neuron')
        plt.colorbar()
        plt.tight_layout()

    incoming_sum = np.sum(surface,axis=0)
    x = np.arange(len(incoming_sum))
    plt.figure(title + ' total incoming connection weights')
    plt.bar(x,incoming_sum)
    if filepath is not None:
        plt.savefig(filepath + '/'+figure+'.eps')

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

def repeat_test_spikes_gen(input_spikes,test_neuron_id,onset_times,test_duration):
    # go through all spikes from onset time -10ms to onset time + 60ms and add this value - the corresponding onset time offset to a new row in a matrix of responses
    # the pre-existing psth function can then be used to plot the output of these collective responses
    import numpy as np
    spikes = input_spikes[test_neuron_id]
    psth_spikes = []
    for i, stimulus in enumerate(onset_times):
        psth_spikes.append([])
        for onset_time in stimulus:
            a = spikes[spikes > onset_time - 10.]
            b = a[a <= onset_time + test_duration]
            c = np.asarray([x.item()- onset_time for x in b])
            psth_spikes[i].append(c)
    return psth_spikes

def sub_pop_builder_inter(sim,post_size,post_type,post_params,pre_type,pre_params,pre_name,
                          post_name,projection_list,sub_pre_pop_size=255.,max_post_per_core=255.,
                          pre_pops=False,post_record_list=["spikes"]):
    import numpy as np
    machine_chip_coordinates = [[0,0],[0,1],[1,0],[1,1]]

    if not isinstance(pre_type,str):
        raise Exception("non spike source array pre pops currently unsupported")
        #TODO: allow for non SSA pre pops to be passed in e.g. SpiNNakEar outputs
    else:
        input_spikes = pre_params
    n_sub_pops = int(np.ceil(post_size / max_post_per_core))

    post_pops =[]
    pre_pop_size = int(len(input_spikes))
    pre_post_projs =[]
    sub_lists=[]
    pres = []
    post_offset = int(max_post_per_core)#post_size / n_sub_pops
    remaining_post_neurons = post_size
    max_pre_index_per_projection=[]
    chip_index = 0

    #create pre pops
    pre_offset = int(sub_pre_pop_size) #pre_pop_size / n_sub_pops

    remaining_pre_neurons = pre_pop_size
    pre_neuron = 0
    #assume even dist of pre to post calculate how many posts per pre
    n_post_steps = len(range(0, post_size, post_offset))
    n_pre_steps = len(range(0,pre_pop_size,pre_offset))

    if pre_pops is False:
        pre_pops =[]
        #evenly split points to create pre pop amongst n_steps
        pre_iterations = np.linspace(0,n_post_steps-1,n_pre_steps,dtype=int)
    else:#setup empty pre_iterations so we don't make new pre pops and calculate the pres list
        pre_iterations = np.asarray([])
        pres=[]
        for pop in pre_pops:
            pres.append(range(pre_neuron, int(pre_neuron + pop.size)))
            pre_neuron += pop.size

    #create post pops
    for i, post_neuron in enumerate(range(0, post_size, post_offset)):

        if post_neuron + post_offset < post_size:
            pop_size = post_offset
        else:
            pop_size = remaining_post_neurons
        post_pops.append(sim.Population(pop_size, post_type, post_params,
                                        label=post_name + "_sub_pop_{}".format(i),
                                        # constraints=[ChipAndCoreConstraint(machine_chip_coordinates[chip_index][0],
                                                                           # machine_chip_coordinates[chip_index][1])]
                                        ))
        post_pops[i].record(post_record_list)
        remaining_post_neurons -= pop_size
        sub_lists.append([(pre,post,weight,delay) for (pre,post,weight,delay) in projection_list if
                    post >= post_neuron and post < (post_neuron + post_offset)])
        #create corresponding pre pops
        if i in pre_iterations:
            for _ in np.nonzero(pre_iterations==i)[0]:
                if _ is not None:
                    if pre_neuron + pre_offset < pre_pop_size:
                        pop_size = pre_offset
                    else:
                        pop_size = remaining_pre_neurons
                    pre_pops.append(
                        sim.Population(pop_size, sim.SpikeSourceArray(spike_times=input_spikes[pre_neuron:pre_neuron+pop_size]),
                                       label=pre_name + "_sub_pop_{}".format(i),
                                        # constraints=[ChipAndCoreConstraint(machine_chip_coordinates[chip_index][0],
                                        #                                    machine_chip_coordinates[chip_index][1])]
                                                                           ))
                    pres.append(range(pre_neuron,int(pre_neuron+pop_size)))
                    remaining_pre_neurons-=pop_size
                    pre_neuron += pre_offset
                    chip_index+=1

    for i, post_neuron in enumerate(range(0, post_size, post_offset)):
        #go through each of the connections and setup relevant projections
        pre_index_store=[]
        pre_lists = [[] for _ in range(len(pres))]
        for (pre,post,weight,delay) in sub_lists[i]:
            #find sub pre pop index
            for idx,ids in enumerate(pres):
                if pre in ids:
                    pre_index = idx
                    if idx not in pre_index_store:
                        pre_index_store.append(idx)
                    break
            # print pre_index
            min_pre = min(pres[pre_index])
            pre_lists[pre_index].append((pre - min_pre,post - post_neuron,weight,delay))
        max_pre_index_per_projection.append(max(pre_index_store)-min(pre_index_store)+1)

        for j,pre_list in enumerate(pre_lists):
            if pre_list is not None and len(pre_list):
                pre_post_projs.append(sim.Projection(pre_pops[j], post_pops[i], sim.FromListConnector(pre_list),
                                              synapse_type=sim.StaticSynapse()))
    return pre_pops,post_pops,pre_post_projs,max_pre_index_per_projection

def sub_pop_projection_builder(pre_pops,post_pops,connection_list,sim,receptor_type='excitatory'):
    import numpy as np
    #used only if the pre and post subpops have already been created
    if pre_pops is None or post_pops is None:
        raise Exception("both input and output populations must already be initialised")

    pres=[]
    posts=[]
    offset = 0.
    for i, pre_pop in enumerate(pre_pops):
        pres.append(np.arange(pre_pop.size)+offset)
        offset = pres[i].max()+1
    offset = 0.
    for i,post_pop in enumerate(post_pops):
        posts.append(np.arange(post_pop.size)+offset)
        offset = posts[i].max()+1
    pres = np.asarray(pres)
    posts = np.asarray(posts)
    pre_post_projs=[]
    post_index=0
    pre_index=0
    for i,post_pop in enumerate(post_pops):
        post_connection_list = [(pre,post,weight,delay) for (pre,post,weight,delay) in connection_list if post in posts[i]]
        pre_lists = [[] for _ in range(len(pres))]
        for (pre,post,weight,delay) in post_connection_list:
            for idx, ids in enumerate(pres):
                if pre in ids:
                    pre_index = idx
                    break

            pre_lists[pre_index].append((pre-pres[pre_index].min(),post-posts[i].min(),weight,delay))
        for j, pre_list in enumerate(pre_lists):
            if pre_list is not None and len(pre_list):
                pre_post_projs.append(sim.Projection(pre_pops[j], post_pop, sim.FromListConnector(pre_list),
                                                     synapse_type=sim.StaticSynapse(),receptor_type=receptor_type))
    return pre_post_projs

def pre_group_generator(input_size,target_pop_size,source_target_list,max_pop_size):
    # go through connectivity list and separate post neurons into sub populations based on common pre neurons
    # build 2D matrix of pre post connectivity
    import numpy as np
    connectivity_matrix = np.zeros((int(input_size), int(target_pop_size)),dtype=bool)
    # weight_matrix = np.zeros((int(input_size), int(target_pop_size)))
    # delay_matrix = np.zeros((int(input_size), int(target_pop_size)))
    weight_matrix = [[[]for __ in range(target_pop_size)]for _ in range(input_size)]
    delay_matrix = [[[]for __ in range(target_pop_size)]for _ in range(input_size)]

    for (pre, post, w, d) in source_target_list:
        if w > 0.:
            connectivity_matrix[int(pre)][int(post)] = 1
            # weight_matrix[int(pre)][int(post)] = w
            # delay_matrix[int(pre)][int(post)] = d
            weight_matrix[int(pre)][int(post)].append(w)
            delay_matrix[int(pre)][int(post)].append(d)
    # group indices of rows of the connectivity matrix that share a percentage of post neurons
    pre_groups = []
    pre_groups_index = 0
    #scramble order of choices for pre neuron to process
    pre_neurons = np.random.choice(input_size,input_size,replace=False)
    # for pre_neuron in range(int(input_size)):
    for pre_neuron in pre_neurons:
        pre_exists = False
        for group in pre_groups:
            if pre_neuron in group:
                pre_exists = True
                break
        #pre_neuron doesn't already exist in a pre_group
        if pre_exists is False:
            post_connections = connectivity_matrix[pre_neuron]
            if post_connections.max()==True:
                pre_groups.append([pre_neuron])
                similarity_matrix = np.sum(connectivity_matrix * post_connections,axis=1)
                #remove matching entry
                similarity_matrix[pre_neuron] = 0
                # group_ids = np.nonzero(similarity_matrix >= similarity_matrix.max() * 0.01)
                #group together pre neurons if the share any post neurons
                group_ids = np.nonzero(similarity_matrix)[0]
                for pre_index in group_ids:
                    # check pre index doesn't already feature in a previous group
                    pre_exists = False
                    for group in pre_groups:
                        if pre_index in group:
                            pre_exists = True
                            break
                    if pre_exists is False:
                        pre_groups[pre_groups_index].append(pre_index)
                pre_groups_index += 1
    # attempt to resort any pre_groups that are too big or small
    for i, group in enumerate(pre_groups):
        if len(group)>max_pop_size:
            split_indices = np.arange(0,len(group),int(max_pop_size))
            for j,fr in enumerate(split_indices):
                if fr == 0:
                    pre_groups[i]=group[:split_indices[j+1]]
                elif (fr + 1000) > len(group):#last split
                    pre_groups.append(group[split_indices[j]:])
                else:
                    pre_groups.append(group[split_indices[j]:split_indices[j+1]])

        elif len(group) == 1:
            pre_neuron = group[0]
            post_connections = connectivity_matrix[pre_neuron]
            similarity_matrix = np.sum(connectivity_matrix * post_connections,axis=1)
            #remove matching entry
            similarity_matrix[pre_neuron] = 0
            max_index = np.argmax(similarity_matrix)
            for j, other_group in enumerate(pre_groups):
                if max_index in other_group:
                    pre_groups[j].append(pre_neuron)
                    break
            del pre_groups[i]

    # pre_id_check = []
    for i,group in enumerate(pre_groups):
        group.sort()
        # for id in group:
            # pre_id_check.append(id)
        pre_groups[i]=np.asarray(group)
    # pre_id_check.sort()
    # if pre_id_check[-1]!=input_size-1:
    #     raise Exception(ArithmeticError("missing pre neurons in subgroups"))
    return pre_groups,connectivity_matrix,np.asarray(weight_matrix),np.asarray(delay_matrix)

def post_group_generator(pre_groups,connectivity_matrix,max_pop_size):
    import numpy as np
    post_groups = []
    mod_pre_groups = []
    mod_pre_group_index=0
    # create post pops
    cum_pre_indices=np.asarray([],dtype=int)
    for i, pre_indices in enumerate(pre_groups):
        # build post groups
        # posts=[]
        # posts= np.nonzero(np.sum(connectivity_matrix[pre_indices],axis=0))[0]
        cum_pre_indices = np.append(cum_pre_indices,pre_indices)
        post_connections = np.sum(connectivity_matrix[pre_indices], axis=0, dtype=bool)
        # cross_over_posts = post_connections* np.sum(np.delete(connectivity_matrix, cum_pre_indices,axis=0),axis=0, dtype=bool)
        # if len(cum_pre_indices)>0:
        #     cross_over_posts = post_connections * np.sum(connectivity_matrix[cum_pre_indices],axis=0, dtype=bool)
        # else:
        #     # cross_over_posts = np.zeros(len(post_connections),dtype=bool)
        cross_over_posts = post_connections * np.sum(np.delete(connectivity_matrix,cum_pre_indices,axis=0), axis=0, dtype=bool)
        posts = np.nonzero(post_connections ^ cross_over_posts)[0]

        if len(posts)>0:
            #calculate how many total pre neurons this group will connect to
            max_n_incoming_spikes = 1000
            n_incoming_spikes = np.count_nonzero(connectivity_matrix[:,np.nonzero(post_connections)[0]])
            if n_incoming_spikes>max_n_incoming_spikes:
                #divide posts into sub_sub_groups!
                group_size = np.nonzero(post_connections)[0].size/(n_incoming_spikes/max_n_incoming_spikes)
            else:
                group_size = max_pop_size

            for posts_idx in np.arange(0,len(posts),int(group_size)):
                if posts_idx + group_size > len(posts):
                    smaller_posts = posts[posts_idx:]
                else:
                    smaller_posts = posts[posts_idx:posts_idx+int(group_size)]

                mod_pre_ids = np.unique(np.nonzero(connectivity_matrix[:,smaller_posts])[0])
                mod_pre_groups.append([])
                for mod_id in mod_pre_ids:
                    pre_exists = False
                    for group in mod_pre_groups:
                        if mod_id in group:
                            pre_exists = True
                            break
                    # pre_neuron doesn't already exist in a pre_group
                    if pre_exists is False:
                        mod_pre_groups[mod_pre_group_index].append(mod_id)
                mod_pre_group_index+=1
                post_groups.append(smaller_posts)

    post_groups = np.asarray([group for group in post_groups if group.size > 0])
    mod_pre_groups = np.asarray([group for group in mod_pre_groups if len(group) > 0])
    return post_groups,mod_pre_groups

def check_incoming_activity(post_neurons,pre_neurons_group,connectivity_matrix,max_n_incoming_spikes):
    import numpy as np
    #create lists of all the pre_neurons that will be transmitting a spike to the post neuron core
    total_incoming_neurons = []
    n_incoming_spikes = 0
    for pre_neurons in pre_neurons_group:
        pre_filter = connectivity_matrix[pre_neurons]
        # n_incoming_spikes += np.sum(pre_filter[:, post_neurons])
        if np.count_nonzero(pre_filter[:, post_neurons]):
            n_incoming_spikes += len(pre_neurons)
            if n_incoming_spikes > max_n_incoming_spikes:
                return False
    return True

def sub_group_generator(pre_size,post_size,projection_list,max_sub_post_pop_size=255.,max_incoming_spikes=2000):
    import numpy as np

    connectivity_matrix = np.zeros((int(pre_size), int(post_size)))#,dtype=bool)
    # weight_matrix = np.zeros((int(input_size), int(target_pop_size)))
    # delay_matrix = np.zeros((int(input_size), int(target_pop_size)))
    weight_matrix = [[[]for __ in range(post_size)]for _ in range(pre_size)]
    delay_matrix = [[[]for __ in range(post_size)]for _ in range(pre_size)]

    for (pre, post, w, d) in projection_list:
        if w > 0.:
            connectivity_matrix[int(pre)][int(post)] += 1
            # weight_matrix[int(pre)][int(post)] = w
            # delay_matrix[int(pre)][int(post)] = d
            weight_matrix[int(pre)][int(post)].append(w)
            delay_matrix[int(pre)][int(post)].append(d)

    pre_groups=[]
    post_group_index = 0
    pre_group_index = 0
    post_neurons = np.random.choice(post_size,post_size,replace=False).tolist()
    filtered_connectivity_matrix = np.copy(connectivity_matrix)
    #minimum number of post groups will equal the total number of machine vertices
    min_n_post_groups = int(np.ceil(post_size/max_sub_post_pop_size))
    post_groups=[[post_neurons.pop(0)] for _ in range(min_n_post_groups)]

    while len(post_neurons)>0:
        if post_group_index==len(post_groups):
            post = post_neurons.pop(0)
        else:
            post = post_groups[post_group_index]
        #find associated pres (that aren't already in a pre group)
        pres = np.nonzero(filtered_connectivity_matrix[:,post])[0]
        #if there are no exclusive pre neurons then add this post to the group with the most shared pre connections?
        if len(pres)==0:
            pre_connections = np.array(connectivity_matrix[:,post],dtype=bool)
            n_shared_pre_connections=[]
            for i,post_group in enumerate(post_groups):
                n_shared_pre_connections.append([np.count_nonzero(np.bitwise_and(pre_connections,
                                                                         np.sum(connectivity_matrix[:,post_group], axis=1,dtype=bool))),i])
            n_shared_pre_connections.sort(reverse=True)
            dummy_post_groups = map(list,post_groups)
            found_candidate = False
            for n_connections,candidate_group_index in n_shared_pre_connections:
                if n_connections > 0:
                    dummy_post_groups[candidate_group_index].append(post)
                    if check_incoming_activity(dummy_post_groups[candidate_group_index], pre_groups,
                                               connectivity_matrix, max_n_incoming_spikes=max_incoming_spikes):
                        post_groups[candidate_group_index].append(post)
                        found_candidate = True
                        break
            if found_candidate==False:
                #no suitable home in other groups so will have to create a new one
                post_groups.append([post])
                post_group_index+=1
        else:
            if post_group_index==len(post_groups):
                post_groups.append([post])
            pre_groups.append(pres.tolist())
            filtered_connectivity_matrix[pre_groups[pre_group_index]] = 0
            # find post neuron that shares the most pres with the initial post neuron in the current group
            potential_post_connections = np.sum(connectivity_matrix[pre_groups[pre_group_index]], axis=0)
            #find max number of post connections (this will probably be the chosen post neurons connections)
            max_n_post_connections = potential_post_connections.max()
            while 1:
                    # remove existing post group matches
                    potential_post_connections[post_groups[post_group_index]] = 0
                    #if no matches then break
                    if potential_post_connections.max()==0:
                    # if potential_post_connections.max()<=max_n_post_connections*0.5:
                        post_group_index += 1
                        pre_group_index += 1
                        break
                    else:
                        max_index = np.argmax(potential_post_connections)
                        # find associated pres that aren't already in the pre list
                        pres = np.nonzero(filtered_connectivity_matrix[:, max_index])[0]
                        if 0:#len(pres)+len(pre_groups[pre_group_index])>max_sub_pre_pop_size:
                            post_group_index += 1
                            pre_group_index += 1
                            break
                        dummy_pre_groups=map(list,pre_groups)
                        for pre in pres:
                            dummy_pre_groups[pre_group_index].append(pre)
                        if check_incoming_activity(post_groups[post_group_index], dummy_pre_groups,
                                                   connectivity_matrix, max_n_incoming_spikes=max_incoming_spikes):
                            for pre in pres:
                                pre_groups[pre_group_index].append(pre)
                            filtered_connectivity_matrix[pre_groups[pre_group_index]] = 0
                            post_groups[post_group_index].append(max_index)
                            #remove from post list
                            try:
                                post_neurons.pop(post_neurons.index(max_index))
                            except ValueError:
                                print("List does not contain value")
                                #find max_index in other post group, remove it and replace with a new value
                                for p_g_index,p_g in enumerate(post_groups):
                                    if max_index in p_g and p_g_index != post_group_index:
                                        del p_g[p_g.index(max_index)]
                                        if len(post_groups[p_g_index])==0:
                                            post_groups[p_g_index].append(post_neurons.pop(0))
                        else:
                            post_group_index+=1
                            pre_group_index+=1
                            break
    post_groups = np.asarray([np.asarray(group) for group in post_groups if len(group) > 0])
    pre_groups = np.asarray([np.asarray(group) for group in pre_groups if len(group) > 0])

    return pre_groups,post_groups,connectivity_matrix,np.asarray(weight_matrix),np.asarray(delay_matrix)


def sub_pop_builder_auto(sim,post_size,post_type,post_params,pre_type,pre_params,pre_name,
                          post_name,projection_list,max_sub_pre_pop_size=255.,max_post_per_core=255.,
                          pre_pops=False,post_record_list=["spikes"]):
    import numpy as np
    if not isinstance(pre_type,str):
        raise Exception("non spike source array pre pops currently unsupported")
        #TODO: allow for non SSA pre pops to be passed in e.g. SpiNNakEar outputs
    else:
        input_spikes = pre_params
    n_sub_pops = int(np.ceil(post_size / max_post_per_core))

    post_pops =[]
    post_pop_dict={}
    posts_from_pop_index_dict={}
    post_pop_index=0
    post_pop_incoming_spike_counts = []
    pre_pop_size = int(len(input_spikes))
    pre_post_projs =[]
    pres = []
    max_pre_index_per_projection=[]
    chip_index = 0
    pre_neuron = 0

    if pre_pops is False:
        pre_pops =[]
    else:#setup empty pre_iterations so we don't make new pre pops and calculate the pres list
        pre_iterations = np.asarray([])
        pres=[]
        for pop in pre_pops:
            pres.append(range(pre_neuron, int(pre_neuron + pop.size)))
            pre_neuron += pop.size
    import time
    # t = time.time()
    # pre_groups,connectivity_matrix,weight_matrix,delay_matrix = pre_group_generator(pre_pop_size, post_size,
    #                                                                                 projection_list,max_sub_pre_pop_size)
    # elapsed_pre_gen_time = time.time() - t
    t = time.time()
    # post_groups,pre_groups = post_group_generator(pre_groups,connectivity_matrix,max_post_per_core)
    pre_groups,post_groups,connectivity_matrix,weight_matrix,delay_matrix = sub_group_generator(pre_pop_size, post_size,
                                                                                    projection_list,max_sub_pre_pop_size)
    elapsed_calc_time = time.time() - t

    t = time.time()
    pre_pop_index=0
    for pre_indices in pre_groups:
        pre_filter = weight_matrix[pre_indices]
        pre_filter_connections = connectivity_matrix[pre_indices]
        #only create pre pop and its associated projections if it actually has nonzero connections to any post neurons
        if pre_filter_connections.max()>0:
            # create pre pop
            pop_size = len(pre_indices)
            sub_spikes = np.asarray(input_spikes)[pre_indices]
            sub_spikes = sub_spikes.tolist()
            pre_pops.append(
                sim.Population(pop_size, sim.SpikeSourceArray(spike_times=sub_spikes),
                               label=pre_name + "_sub_pop_{}".format(pre_pop_index),
                               # constraints=[ChipAndCoreConstraint(machine_chip_coordinates[chip_index][0],
                               #                                    machine_chip_coordinates[chip_index][1])]
                               ))

            # find connected post groups
            connected_post_group_indices = []
            for posts in post_groups:
                pre_post_filter = pre_filter[:, posts]
                if np.count_nonzero(pre_post_filter)>0:
                    pop_size=len(posts)
                    #if post pop not previously created
                    if str(posts) not in post_pop_dict:
                        post_pops.append(sim.Population(pop_size, post_type, post_params,
                                                        label=post_name + "_sub_pop_{}".format(post_pop_index),
                                                        # constraints=[ChipAndCoreConstraint(machine_chip_coordinates[chip_index][0],
                                                                                           # machine_chip_coordinates[chip_index][1])]
                                                        ))
                        post_pops[post_pop_index].record(post_record_list)
                        post_pop_dict[str(posts)] = post_pop_index
                        posts_from_pop_index_dict[str(post_pop_index)] = posts
                        post_pop_index+=1
                        post_pop_incoming_spike_counts.append(0)

                    sub_lists=[]
                    # pre_post_filter = pre_filter[:, posts]
                    delay_filter = delay_matrix[pre_indices]
                    pre_post_delay_filter = delay_filter[:,posts]
                    for pre_index,weights in enumerate(pre_post_filter):
                        if np.count_nonzero(weights)>0:
                            post_indices = np.nonzero(weights)[0]
                            for post_index in post_indices:
                                m_weights=weights[post_index]
                                for i,weight in enumerate(m_weights):
                                    sub_lists.append((pre_index,post_index,weight,pre_post_delay_filter[pre_index][post_index][i]))

                    pre_post_projs.append(sim.Projection(pre_pops[pre_pop_index], post_pops[post_pop_dict[str(posts)]], sim.FromListConnector(sub_lists),
                                                         synapse_type=sim.StaticSynapse()))
                    post_pop_incoming_spike_counts[post_pop_dict[str(posts)]]+=pre_pops[pre_pop_index].size
            pre_pop_index+=1
        chip_index+=1
    elapsed_pop_gen_projection_time = time.time() - t
    print "sub pop building complete, n_sub_pre_pops={}, n_sub_post_pops={}".format(len(pre_pops),len(post_pops))
    print "sub pop build times: sub_pop_calc={}, sub_pop_gen_project={} ".format(elapsed_calc_time,elapsed_pop_gen_projection_time)
    print "post pop incoming spike counts: {}".format(post_pop_incoming_spike_counts)
    return pre_pops,post_pops,pre_post_projs,max_pre_index_per_projection,posts_from_pop_index_dict

def get_sub_pop_spikes(pops,posts_from_pop_index_dict=None):
    import numpy as np
    if posts_from_pop_index_dict is None:
        output_spikes = []
        for pop in pops:
            data = pop.get_data(["spikes"])
            spikes = data.segments[0].spiketrains
            for neuron in spikes:
                output_spikes.append(neuron)
    else:
        total_pop_size = 0
        for pop in pops:
            total_pop_size+=pop.size
        output_spikes=[[] for _ in range(int(total_pop_size))]
        # output_spikes=np.array()
        for i,pop in enumerate(pops):
            data = pop.get_data(["spikes"])
            spikes = data.segments[0].spiketrains
            neuron_indices = posts_from_pop_index_dict[str(i)]
            for j,neuron in enumerate(spikes):
                output_spikes[neuron_indices[j]] = neuron
    return output_spikes

