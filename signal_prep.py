import numpy
import math
from scipy.io import wavfile
# import scipy.signal
from nnresample import resample
from matplotlib.ticker import FormatStrFormatter


def generate_signal(signal_type="tone", fs=22050., dBSPL=40.,
                    freq=3000., duration=0.5, ramp_duration=0.003,
                    silence_duration=0.05, modulation_freq=0.,
                    modulation_depth=1., plt=None, file_name=None,
                    silence=True, title='', ascending=True, channel=0,
                    max_val=None):
    T = 1./fs
    num_samples = int(fs * duration)
    if signal_type == "tone":
        map_bs_cos_shift = numpy.pi / 2

        if modulation_freq > 0:
            signal = [(numpy.sin(2 * numpy.pi * freq * T * i +
                                 map_bs_cos_shift)) *
                      (0.5 * (1. + modulation_depth * numpy.cos(
                          2 * numpy.pi * modulation_freq * T * i)))
                      for i in range(int(num_samples))]
        else:
            signal = [(numpy.sin(2 * numpy.pi * freq * T * i +
                                 map_bs_cos_shift))
                      for i in range(int(num_samples))]
    elif signal_type == "sweep_tone":
        if len(freq) < 2:
            print "enter low and high frequency " \
                  "sweep values as freq=[low,high]"
        phi = 0
        f = freq[0]
        delta = 2. * numpy.pi * f * T
        f_delta = numpy.power((freq[1] / freq[0]), 1. / num_samples)
        signal = []
        for i in range(int(num_samples)):
            signal.append(numpy.sin(phi))
            phi = phi + delta
            f = f * f_delta
            # f = f + f_delta
            delta = 2. * numpy.pi * f * T
    if not ascending:
        signal = signal[::-1]
    elif signal_type == "file":
        if file_name:
            [fs_f, signal] = wavfile.read(file_name)
            signal = numpy.copy(signal)
            if len(signal.shape) > 1:  # stereo
                signal = signal[:, channel]
            fs_f = numpy.float64(fs_f)
            if fs_f != fs:
                signal = resample(signal, fs, fs_f)
        else:
            raise Exception("must include valid wav filename")
    elif signal_type == 'noise':
        if modulation_freq > 0:
            signal = [(2 * (numpy.random.rand() - 0.5)) *
                      (0.5 * (1. + modulation_depth *
                              numpy.cos(2 * numpy.pi *
                                        modulation_freq * T * i)))
                      for i in range(int(num_samples))]
        else:
            signal = [(2 * (numpy.random.rand() - 0.5))
                      for i in range(int(num_samples))]
    elif signal_type == 'pink_noise':
        import colorednoise as cn
        signal = cn.powerlaw_psd_gaussian(1, int(num_samples))
    elif signal_type == 'click':
        signal = [1. for i in range(int(num_samples))]
    else:
        print "invalid signal type!"
        signal = []
    # map_bs_remove 1st sample!?
    # signal = signal[1:]
    # amplitude conversion
    signal = numpy.float64(signal)
    target_rms = 1. * 28e-6 * 10. ** (dBSPL / 20.)
    if max_val is None:
        max_val = numpy.mean(signal ** 2) ** 0.5
    amp = target_rms / max_val
    signal *= -amp  # set loudness

    # add ramps
    num_ramp_samples = numpy.round(fs * ramp_duration)
    step = numpy.pi / (num_ramp_samples - 1)

    for i in range(int(num_ramp_samples)):
        ramp = (1 + numpy.cos(numpy.pi + i * step)) / 2.
        # on ramp
        signal[i] *= ramp
        # off ramp
        signal[-i] *= ramp
    if silence:
        # add silence
        num_silence_samples = int(numpy.ceil(fs*silence_duration))
        # silence is realistic noise -20dBSPL
        silence_amp = 1. * 28e-6 * 10. ** (-20. / 20.)
        silence_samples = ((2 * numpy.random.rand(num_silence_samples))
                           - 1.) * silence_amp
        signal_silence_samples = ((2 * numpy.random.rand(len(signal)))
                                  - 1.) * silence_amp
        signal = numpy.concatenate((silence_samples, signal +
                                    signal_silence_samples, silence_samples))
    else:
        signal = numpy.asarray(signal)
    if plt:
        plt.figure(title)
        time = numpy.linspace(0, len(signal) / fs, len(signal))
        plt.plot(time, signal)
        plt.xlim((0, len(signal) / fs))
        plt.xlabel("time (s)")
        plt.ylabel("signal amplitude")
    return signal


def generate_psth_8(target_neuron_ids, spike_trains, bin_width,
                    duration):
    import numpy as np
    bins = np.arange(bin_width * 1000., duration * 1000., bin_width * 1000.)
    if isinstance(spike_trains, list):
        spike_trains = np.asarray(spike_trains)
    target_neurons = spike_trains[target_neuron_ids]
    hist = []
    for spike_times in target_neurons:
        hist.append(np.histogram(spike_times, bins=bins)[0])
    output = np.mean(hist, axis=0) * 1. / bin_width
    return output


def psth_plot_8(plt, target_neuron_ids, spike_trains, bin_width, duration,
                title='PSTH', filepath=None, subplots=None, file_format='pdf',
                file_name='', ylim=None):
    PSTH = generate_psth_8(target_neuron_ids, spike_trains,
                           bin_width=bin_width, duration=duration)
    x = numpy.linspace(0, duration, len(PSTH))
    if subplots is None:
        plt.figure(title)
        plt.xlabel("time (s)")
        plt.ylabel("firing rate (sp/s)")
        plt.plot(x, PSTH)
        # plt.bar(x, PSTH)
    else:
        if subplots[2] == 1:
            plt.gcf().text(0.04, 0.5, 'firing rate (sp/s)', va='center',
                           rotation='vertical')
        ax = plt.subplot(subplots[0], subplots[1], subplots[2])
        ax.plot(x, PSTH)
        if ylim is not None:
            ax.set_ylim((0, ylim))
        ax2 = ax.twinx()
        ax2.set_ylabel(title)
        ax2.set_yticks([])
        # ax.set_title(title)
        if subplots[2] == subplots[0]:
            ax.set_xlabel("time (s)")
        else:
            ax.set_xticklabels([])

    plt.xlim((0, duration))

    if ylim is None:
        max_rate = max(PSTH)
        plt.ylim((0, max_rate + 1))
    # else:
    #     plt.ylim((0,ylim))
    if filepath is not None:
        if subplots is None or subplots[2] == subplots[0]:
            plt.savefig(filepath + '/' + file_name + '{}.'.format(title)
                        + file_format)
    return PSTH


def spike_raster_plot_8(spikes, plt, duration, ylim, scale_factor=0.001,
                        title=None, filepath=None, file_format='pdf',
                        file_name='', xlim=None, onset_times=None,
                        pattern_duration=None, markersize=3,
                        marker_colour='black', alpha=1., subplots=None,
                        legend_strings=None, ylims=None):
    if len(spikes) > 0:
        spike_ids = []
        spike_times = []

        for neuron_index, times in enumerate(spikes):
            for time in times:
                spike_ids.append(neuron_index + 1)
                spike_times.append(time)
        scaled_times = [spike_time * scale_factor
                        for spike_time in spike_times]

        if subplots is None:
            plt.figure(title)
            plt.xlabel("time (s)")
            plt.plot(scaled_times, spike_ids, '.', markersize=markersize,
                     markerfacecolor=marker_colour, markeredgecolor='none',
                     markeredgewidth=0, alpha=alpha)
            plt.ylabel("neuron ID")

        else:
            ax = plt.subplot(subplots[0], subplots[1], subplots[2])
            if 0:  # subplots[2]==1:
                plt.gcf().text(0.04, 0.5, 'repetition', va='center',
                               rotation='vertical')
                plt.gcf().text(0.04, 0.5, 'neuron ID', va='center',
                               rotation='vertical')

            if title is not None:
                # ax.set_title(title)
                ax2 = ax.twinx()
                ax2.set_ylabel(title)
                ax2.set_yticks([])
            if subplots[2] == subplots[0]:
                ax.set_xlabel("time (s)")
            else:
                ax.set_xticklabels([])
            ax.plot(scaled_times, spike_ids, '.', markersize=markersize,
                    markerfacecolor=marker_colour, markeredgecolor='none',
                    markeredgewidth=0, alpha=alpha)
            ax.set_ylabel("neuron ID")

        if ylims is None:
            plt.ylim(0, ylim)
        else:
            plt.ylim(ylims)
        plt.xlim(0, duration)

        if onset_times is not None:
            # plot block of translucent colour per pattern
            ax = plt.gca()
            pattern_legend = []
            legend_labels = []
            colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
            # labels = ['A','B','C',]
            for i, pattern in enumerate(onset_times):
                pattern_legend.append(plt.Line2D([0], [0],
                                                 color=colours[i % 8],
                                                 lw=4, alpha=0.2))
                legend_labels.append("s{}".format(i+1))
                for onset in pattern:
                    x_block = (onset, onset+scale_factor * pattern_duration)
                    ax.fill_between(x_block, ylim, alpha=0.2,
                                    facecolor=colours[i % 8], lw=0.5)
            plt.legend(pattern_legend, legend_labels,
                       bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=len(onset_times), mode="expand", borderaxespad=0.)
        if xlim is not None:
            plt.xlim(xlim)
        if legend_strings is not None and (subplots is None
                                           or subplots[2] == 1):
            plt.legend(legend_strings, bbox_to_anchor=(0.1, 1.25),
                       loc='upper center', ncol=len(legend_strings),
                       markerscale=10.)
        if filepath is not None:
            if subplots is None or subplots[2] == subplots[0]:
                plt.savefig(filepath + '/' + file_name + '{}.'.format(title)
                            + file_format)


def abr_spikes(neuron_times, duration_ms):
    import numpy as np
    abr = np.zeros(int((duration_ms) / 0.1))
    x = np.arange(0, (duration_ms), 0.1)
    flat_spikes = []
    for neuron in neuron_times:
        for time in neuron:
            flat_spikes.append(time.item())
    flat_spikes = np.asarray(flat_spikes)
    unique_times = np.unique(flat_spikes)
    for time in unique_times:
        closest_index = int((np.abs(x - time)).argmin())
        abr[closest_index] += len(np.where(flat_spikes == time)[0])
    return abr, x


def abr_mem_v(v, duration_ms, ref_v=-100.):
    import numpy as np
    v = np.asarray(v[0])
    x = np.linspace(0, (duration_ms), len(v))
    # reference_voltage = ref_v * pq.mV
    abr = np.sum(v - ref_v, axis=1)
    return abr, x


def multi_spike_raster_plot(spikes_list, plt, duration, ylim,
                            scale_factor=0.001, marker_size=3,
                            dopamine_spikes=[], title=''):
    plt.figure(title)
    marker_colours = ['black', 'blue', 'red', 'green']
    marker_styles = ['.', '+', '*', 'o']
    count = 0
    for spikes in spikes_list:
        if len(spikes) > 0:
            spike_times = [spike_time for (neuron_id, spike_time) in spikes]
            scaled_times = [spike_time * scale_factor
                            for spike_time in spike_times]
            spike_ids = [neuron_id for (neuron_id, spike_time) in spikes]
            spike_ids[:] = [neuron_id + 1 for neuron_id in spike_ids]

            plt.plot(scaled_times, spike_ids, marker_styles[count],
                     markersize=marker_size, color=marker_colours[count])
            plt.ylim(0, ylim)
            plt.xlim(0, duration)
        count += 1
    if len(dopamine_spikes) > 0:
        spike_times = [spike_time
                       for (neuron_id, spike_time) in dopamine_spikes]
        scaled_times = [spike_time * scale_factor
                        for spike_time in spike_times]
        for xc in scaled_times:
            plt.axvline(x=xc, color='red')
        plt.ylim(0, ylim)
        plt.xlim(0, duration)


def weight_array_to_group_list(weight_array, from_ids, to_ids):
    group_weights_to = []
    group_weights_from = []
    for id in from_ids:  # group_target_ids:
        connection_weights_from = [weight for weight in weight_array[id][:]
                                   if not math.isnan(weight)]
        group_weights_from.append(numpy.array(connection_weights_from))
    for id in to_ids:
        connection_weights_to = [weight for weight in weight_array[:][id]
                                 if not math.isnan(weight)]
        group_weights_to.append(numpy.array(connection_weights_to))
    return [group_weights_to, group_weights_from]


def vary_weight_plot(varying_weights, ids, stim_ids, duration, plt, num_recs,
                     np, ylim, title='', filepath=None, legend=None,
                     figsize=None):
    if len(varying_weights) != num_recs:
        raise Exception("incorrect number of weight recordings taken "
                        "(num_recs={}, len(varyingweights={})"
                        .format(num_recs, len(varying_weights)))
    if len(ids) > 0:
        repeats = np.linspace(0, duration, num_recs)
        sr = math.sqrt(len(ids))
        num_cols = np.ceil(sr)
        num_rows = np.ceil(len(ids) / num_cols)
        if num_rows > 1 and num_cols > 1 and figsize is None:
            figsize = (5 * num_cols, 3 * num_rows)
        elif figsize is None:
            figsize = (8, 6)
        else:
            figsize = figsize
        if stim_ids:
            plt.figure(title, figsize=figsize)
            plt.suptitle("non-pattern synapse weight updates for post neurons")
            plt.figure(title + "pattern", figsize=figsize)
            plt.suptitle("pattern synapse weight updates for post neurons")
        else:
            plt.figure(title, figsize=figsize)
            plt.suptitle("synapse weight updates for post neurons")

        count = 0
        for id in ids:
            if legend is not None:
                legend_string = []
                legend_line = []
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
                            legend_string.append("pre_id:{}".format(pre + 1))
                rec_index += 1

            plt.figure(title)
            plt.subplot(num_rows, num_cols, count + 1)
            id_times = numpy.array(id_times)
            if len(id_times.shape) > 1:
                id_times = map(list, zip(*id_times))
                for t in id_times:
                    line, = plt.plot(repeats, t)
                    if legend is not None:
                        legend_line.append(plt.Line2D([0], [0],
                                                      color=line.get_color(),
                                                      lw=4))
                if legend is not None:
                    plt.legend(legend_line, legend_string,
                               bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                               ncol=3, mode="expand", borderaxespad=0.)
                plt.ylabel("ID:{}".format(str(id+1)))
                plt.xticks(np.linspace(0, duration, 5))
                if num_rows > 2:
                    plt.gca().yaxis.set_major_formatter(FormatStrFormatter
                                                        ('%.1f'))
                    plt.yticks(np.linspace(0, ylim, 3))
                else:
                    plt.xlabel('time (s)')
                plt.xlim(0, duration)
                plt.ylim(0, ylim)
            if id_times_pattern:
                plt.figure(title + "pattern")
                plt.subplot(num_rows, num_cols, count + 1)
                id_times_pattern = numpy.array(id_times_pattern)
                if len(id_times_pattern.shape) > 1:
                    id_times_pattern = map(list, zip(*id_times_pattern))
                    for t in id_times_pattern:
                        plt.plot(repeats, t)
                plt.ylabel("ID:{}".format(str(id + 1)))
                plt.xticks(np.linspace(0, duration, 5))
                if num_rows > 2:
                    plt.gca().yaxis.set_major_formatter(FormatStrFormatter
                                                        ('%.1f'))
                    plt.yticks(np.linspace(0, ylim, 3))
                else:
                    plt.xlabel('time (s)')
                plt.xlim(0, duration)
                plt.ylim(0, ylim)
            # else:
            #    plt.plot(repeats,id_times)
            count += 1
        if filepath is not None:
            plt.figure(title)
            plt.savefig(filepath+'/{}_weights.eps'.format(title))
            if stim_ids:
                plt.figure(title + "pattern")
                plt.savefig(filepath+'/{}_pattern_weights.eps'.format(title))


def weight_dist_plot(varying_weights, num_ticks, plt, w_min, w_max, np=numpy,
                     title=None, filepath=None):
    # assumes varying weights is a list of connectionsholders,
    # for a distribution plot we don't care about the ids
    v_weights = []
    for neuron, reading in enumerate(varying_weights):
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
    plt.hist(init_weights, bins=100, alpha=0.5, range=(w_min, w_max * 1.1))
    plt.hist(fin_weights, bins=100, alpha=0.5, range=(w_min, w_max * 1.1))
    plt.legend(["first recording", "last recording"])
    plt.ylabel("number of synapses")
    plt.xlabel("weight of synapse")
    if filepath is not None:
        plt.savefig(filepath + '/' + title + '.pdf')


def cell_voltage_plot_8(v, plt, duration_ms, time_step_ms, scale_factor=0.001,
                        id=None, title='', filepath=None, subplots=None):
    # times = range(0,int(duration_ms),int(time_step_ms))
    membrane_voltage = v[0]
    times = range(0, membrane_voltage.shape[0])
    scaled_times = [time * scale_factor for time in times]
    if id is not None:
        mem_v = [v_t[id] for v_t in membrane_voltage]
        if isinstance(id, float):
            title = title + str(id + 1)
    else:
        mem_v = membrane_voltage
        # title = title + "{} neurons".format(membrane_voltage.shape[1])
    if subplots is None:
        plt.figure(title)
    else:
        ax = plt.subplot(subplots[0], subplots[1], subplots[2])
        ax.set_title(title)
    plt.plot(scaled_times, mem_v)
    plt.xlim((0, duration_ms * 0.001))
    if subplots is None:
        plt.xlabel('time (s)')
        plt.ylabel('membrane voltage (mV)')
    else:
        if subplots[2] == subplots[0]:
            plt.xlabel("time (s)")
    if filepath is not None:
        plt.savefig(filepath + '/' + title + '_memV.eps')


def cell_voltage_plot(v, plt, duration, scale_factor=0.001, id=0, title=''):
    times = [i[1] for i in v if i[0] == id]
    scaled_times = [time * scale_factor for time in times]
    membrane_voltage = [i[2] for i in v if i[0] == id]
    plt.figure(title + str(id + 1))
    plt.plot(scaled_times, membrane_voltage)


def fixed_prob_connector(num_pre, num_post, p_connect, weights, delays=1.,
                         self_connections=False, single_pop=False):
    connection_list = []
    # do pre->post connections
    num_connections_per_neuron = int(num_post * p_connect)
    for id in range(num_pre):
        choice = range(num_post)
        if single_pop:
            # exclude current id from being chosen
            choice.remove(id)
        connections = numpy.random.choice(choice, num_connections_per_neuron,
                                          replace=False)
        for target_id in connections:
            connection_list.append((id, target_id, weights, delays))

    if not single_pop:
        # do post->pre connections
        num_connections_per_neuron = num_pre * p_connect
        for id in range(num_post):
            choice = range(num_pre)
            if single_pop:
                # exclude current id from being chosen
                choice.remove(id)
            connections = numpy.random.choice(choice,
                                              num_connections_per_neuron,
                                              replace=False)
            for target_id in connections:
                connection_list.append((id, target_id, weights, delays))

    return connection_list


def find_nearest(array, value):
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# distance dependent connectivity generator using a normal probability
#  distribution
def normal_dist_connection_builder(pre_size, post_size, RandomDistribution,
                                   conn_num, dist, sigma, conn_weight=None,
                                   delay=1., p_connect=1.0, delay_scale=None,
                                   dist_weight=None, posts=None,
                                   multapses=True, normalised_space=None,
                                   get_max_dist=False):
    import numpy as np
    if posts is None:
        posts = xrange(post_size)
    conn_list = []
    if normalised_space is None:
        if pre_size > post_size:
            post_scale = float(pre_size - 1) / (post_size - 1)
            pre_scale = 1.
        else:
            post_scale = 1.
            pre_scale = float(post_size - 1) / (pre_size - 1)
    else:
        post_scale = float(normalised_space - 1) / (post_size - 1)
        pre_scale = float(normalised_space - 1) / (pre_size - 1)

    if get_max_dist is True:
        max_dist = 0
    for post in posts:
        scaled_post = int(post * post_scale)
        mu = int(dist / 2) + scaled_post * dist
        pre_dist = RandomDistribution('normal_clipped', [mu, sigma, 0,
                                                         pre_size *
                                                         pre_scale - 1])
        if isinstance(conn_num, float) or isinstance(conn_num, int):
            number_of_connections = conn_num
        else:
            number_of_connections = conn_num.next(n=1)
        pre_idxs = pre_dist.next(n=int(number_of_connections + 0.5))
        if multapses is False:
            samples = np.unique(np.array(pre_idxs, dtype=int))
            loop_count = 0
            while len(samples) < int(number_of_connections) and loop_count < 2:
                samples = np.unique(np.append(samples,
                                              np.array(pre_dist.next(n=1000),
                                                       dtype=int)))
                loop_count += 1
            pre_idxs = np.random.choice(samples, int(number_of_connections),
                                        replace=False)

        for pre in pre_idxs:
            scaled_pre = int(np.round(pre / pre_scale))
            if get_max_dist is True:
                if abs(pre-mu) > max_dist:
                    max_dist = abs(pre - mu)
            if np.random.rand() <= p_connect:
                if conn_weight is None:
                    conn_list.append((scaled_pre, post))
                else:
                    if type(conn_weight) == float:
                        weight = conn_weight
                    elif dist_weight is not None:
                        # p_dist = np.exp(float(abs(scaled_pre - post)) /
                        # (post_size * post_scale) - 1.)
                        p_dist = float(abs(scaled_pre - post)) / \
                                 (post_size * post_scale)
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
    if get_max_dist is True:
        return conn_list, max_dist
    else:
        return conn_list


# find stimulus onset times from audio signal
def audio_stimulus_onset_detector(audio_signal, Fs, num_classes):
    import numpy as np
    # average samples over previous 10ms absolute values
    num_10ms_samples = Fs * 0.01
    envelope_upper_threshold = 0.00005
    envelope_lower_threshold = 0.00002
    stimulus_times = []
    for j in range(num_classes):
        stimulus_times.append([])
    class_index = 0
    triggered = False
    for i, sample in enumerate(audio_signal):
        if i >= num_10ms_samples:
            envelope = np.mean(np.absolute(audio_signal
                                           [int(i - num_10ms_samples):i]))
            if envelope > envelope_upper_threshold and triggered is False:
                stimulus_times[class_index].append(i / Fs)
                if class_index < (num_classes - 1):
                    class_index += 1
                else:
                    class_index = 0
                triggered = True
            if envelope < envelope_lower_threshold and triggered is True:
                triggered = False

    return stimulus_times


# stimulus onset times is a list of onset times lists for each stimulus
# spike time is the output spikes from a population
def neuron_correlation(spike_train, time_window, stimulus_onset_times, max_id,
                       noise_threshold, np=numpy,
                       significant_spike_count=None):
    correlations = []  # 1st dimension is stimulus class
    # #counts = np.asarray([np.zeros(max_id + 1), np.zeros(max_id + 1)])
    counts = []
    for stimulus_index, stimulus in enumerate(stimulus_onset_times):
        correlations.append([[] for _ in range(int(max_id + 1))])
        counts.append(np.zeros(max_id + 1))
        # counts.append(np.zeros(max_id + 1))
        # loop through each stimulus onset time and check all neuron firing
        # times
        # save neuron index if firing is within time window of stimulus onset
        for t_index, time in enumerate(stimulus):
            for neuron in correlations[stimulus_index]:
                neuron.append(0)
            for (neuron_id, spike_time) in spike_train:
                if spike_time > time and spike_time <= (time + time_window):
                    correlations[stimulus_index][int(neuron_id)][t_index] += 1

        # check which neurons have fired above or below the noise threshold
        for id, neuron in enumerate(np.asarray(correlations[stimulus_index])):
            # generate count of neuron firings that are above the noise
            # threshold
            counts[stimulus_index][id] = np.count_nonzero(neuron >
                                                          noise_threshold)

    if significant_spike_count is None:
        significant_spike_count = np.mean(counts, axis=1)

    selective_neuron_ids = selective_id_finder(counts, significant_spike_count)

    return np.asarray(counts), selective_neuron_ids, significant_spike_count


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


def selective_neuron_search(pattern_spikes, spike_train, time_window,
                            final_pattern_start, plt, filepath=None, np=numpy,
                            significant_spike_count=None):
    # take final 10% of pattern spikes
    stimulus_times = []
    for pattern in pattern_spikes:
        stimulus_times.append([time for time in pattern
                               if time >= final_pattern_start])
    final_spike_train = []
    for train in spike_train:
        final_spike_train.append([time for time in train
                                  if time >= final_pattern_start])

    max_id = len(spike_train)
    counts, selective_neuron_ids, significant_spike_count = \
        neuron_correlation(final_spike_train, time_window, stimulus_times,
                           max_id,
                           significant_spike_count=significant_spike_count)

    print "significant spike count: {}".format(significant_spike_count)
    max_count = counts.max()
    plt.figure(figsize=(20, 10))
    title = "{}ms post-stimulus spike count for target layer"\
        .format(time_window)
    plt.title(title)
    plt.xlabel("neuron ID")
    plt.ylabel("spike count")
    plt.plot(counts.T)
    legend_string = []
    for i in range(len(stimulus_times)):
        legend_string.append("stimulus {}".format(i + 1))
    plt.legend(legend_string)
    plt.ylim((0, max_count + 1))

    for i in range(len(selective_neuron_ids)):
        print selective_neuron_ids[i]

    if filepath is not None:
        plt.savefig(filepath + "/{}.eps".format(title))
        import csv
        from itertools import izip_longest
        with open(filepath + "/selective_neurons.csv", "w+") as f:
            writer = csv.writer(f)
            for values in izip_longest(*selective_neuron_ids):
                writer.writerow(values)


# assumes input connectivity in varying_weights format
def connection_hist_plot(varying_weights, pre_size, post_size, plt, title='',
                         filepath=None, weight_min=0.000001):
    incoming_connections = [[]for _ in range(post_size)]
    source_list = []
    target_list = []
    # take final reading
    final_connections = varying_weights[-1]
    for (source, target, weight) in final_connections:
        if source is not None and weight > weight_min:
            if source in incoming_connections[int(target)]:
                print "multapse detected!"
            incoming_connections[int(target)].append(source)
            source_list.append(source)
            target_list.append(target)
    out_figure = title + 'pre_pop outgoing connections'
    plt.figure(out_figure)
    plt.hist(source_list, bins=pre_size, alpha=0.5, range=(0, pre_size))
    if filepath is not None:
        plt.savefig(filepath + '/' + out_figure + '.eps')
    in_figure = title + 'post_pop incoming connections'
    plt.figure(in_figure)
    plt.hist(target_list, bins=post_size, alpha=0.5, range=(0, post_size))
    if filepath is not None:
        plt.savefig(filepath + '/' + in_figure + '.eps')


def connection_surface_plot(varying_weights, pre_size, post_size, plt,
                            title='', filepath=None, n_plots=2):
    import numpy as np
    incoming_connections = [[]for _ in range(post_size)]
    plot_indices = [int(idx) for idx in np.linspace(0,
                                                    len(varying_weights) - 1,
                                                    n_plots)]
    for i in plot_indices:
        final_connections = varying_weights[i]
        surface = np.zeros((pre_size, post_size))
        for (source, target, weight) in final_connections:
            if source in incoming_connections[int(target)]:
                print "multapse detected!"
            incoming_connections[int(target)].append(source)
            surface[source][target] += weight

        figure = title + ' connection weights {}'.format(i)
        plt.figure(figure)
        plt.imshow(surface, vmin=0, vmax=surface.max(), interpolation='none',
                   origin='lower', cmap='viridis')
        plt.xlabel('target neuron')
        plt.ylabel('source neuron')
        plt.colorbar()
        plt.tight_layout()

    incoming_sum = np.sum(surface, axis=0)
    x = np.arange(len(incoming_sum))
    plt.figure(title + ' total incoming connection weights')
    plt.bar(x, incoming_sum)
    if filepath is not None:
        plt.savefig(filepath + '/' + figure + '.eps')


def sparsity_measure(onset_times, output_spikes, onset_window=5., from_time=0):
    import numpy as np
    n_neurons = float(len(output_spikes))
    sparsity_matrix = [[] for _ in range(len(onset_times))]
    # np_output_spikes = [[0.]for _ in range(int(n_neurons))]
    # for id, neuron in enumerate(output_spikes):
    #     for spike in neuron:
    #         np_output_spikes[id].append(spike.item())
    # np_output_spikes = np.asarray(np_output_spikes)

    # go through each stimulus onset time and bin all the subsequent output
    # spike IDs that fall in onset time + onset window
    for id, stimulus in enumerate(onset_times):
        for time in stimulus:
            if time >= from_time:
                counts = np.zeros((int(n_neurons), int(onset_window)))
                for out_id, neuron in enumerate(output_spikes):
                    for output_spike in neuron:
                        # only care if at least one spike per neuron has
                        # occured in window
                        if output_spike >= time and output_spike < \
                                (time+onset_window):  # and counts[out_id]==0.:
                            if isinstance(output_spike, (int, float)):
                                counts[out_id, int(output_spike-time-1)] += 1
                            else:  # assume quantity
                                counts[out_id, int(output_spike.item() -
                                                   time - 1)] += 1
                # calculate sum of active neurons across presentation
                # sum =  np.sum(np.sum(counts,axis=0))
                sum = np.sum(counts, axis=1)
                # average across timesteps in window
                av = np.mean(sum)
                sparsity_matrix[id].append(av)
    return sparsity_matrix


def repeat_test_spikes_gen(input_spikes, test_neuron_id, onset_times,
                           test_duration_ms, pre_ms=20.):
    # go through all spikes from onset time -10ms to onset time + 60ms
    # and add this value - the corresponding onset time offset to a new row
    # in a matrix of responses
    # the pre-existing psth function can then be used to plot the output of
    # these collective responses
    import numpy as np
    spikes = input_spikes[test_neuron_id]
    psth_spikes = []
    for i, stimulus in enumerate(onset_times):
        psth_spikes.append([])
        for onset_time in stimulus:
            a = spikes[spikes > (onset_time - pre_ms)]
            b = a[a <= onset_time + test_duration_ms]
            c = np.asarray([x.item() - (onset_time - pre_ms) for x in b])
            psth_spikes[i].append(c)
    return psth_spikes


def fixed_p_connection_builder(pre_size, post_size, p_connect):
    conn_list = []
    import numpy as np
    for post in range(int(post_size)):
        for pre in range(int(pre_size)):
            if np.random.rand() < p_connect:
                conn_list.append((pre, post))

    return conn_list


# assumes binaural data format
def split_population_data_combine(split_data, variable_list):
    import numpy as np
    spikes_combined = []
    mem_v_combined = []

    variable_dict = {
        'spikes': spikes_combined,
        'v': mem_v_combined
    }
    for stack in split_data:
        import quantities as nq
        if 'spikes' in variable_list:
            split_spikes = [split.segments[0].spiketrains for split in stack]
            spikes_combined.append([val for tup in zip(*split_spikes)
                                    for val in tup])

        if 'v' in variable_list:
            mem_v_split = [split.segments[0].filter(name='v')[0]
                           for split in stack]
            sum_width = sum([v.shape[-1] for v in mem_v_split])
            step = len(mem_v_split)
            mem_v = np.empty((len(mem_v_split[0]), sum_width)) * nq.mV
            for i, v in enumerate(mem_v_split):
                mem_v[:, i::step] = v
            # wrapped in a list to look like a usual mem_v `analog signal'
            mem_v_combined.append([mem_v])

    return variable_dict


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
