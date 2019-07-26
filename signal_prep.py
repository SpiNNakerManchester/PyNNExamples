

def spike_raster_plot_8(
        spikes, plt, duration, ylim, scale_factor=0.001, title=None,
        filepath=None, file_format='pdf', file_name='', xlim=None,
        onset_times=None, pattern_duration=None, markersize=3,
        marker_colour='black', alpha=1., subplots=None, legend_strings=None):

    if len(spikes) > 0:
        neuron_index = 1
        spike_ids = []
        spike_times = []

        for times in spikes:
            for time in times:
                spike_ids.append(neuron_index)
                spike_times.append(time)
            neuron_index += 1
        scaled_times = [spike_time * scale_factor for spike_time in spike_times]

        ##plot results
        if subplots is None:
            plt.figure(title)
            plt.xlabel("time (s)")
        else:
            ax = plt.subplot(subplots[0], subplots[1], subplots[2])
            ax.set_title(title)
            if subplots[2]==subplots[0]:
                plt.xlabel("time (s)")
            else:
                ax.set_xticklabels([])
        plt.plot(scaled_times, spike_ids, '.', markersize=markersize,
                 markerfacecolor=marker_colour, markeredgecolor='none',
                 markeredgewidth=0,alpha=alpha)
        plt.ylim(0, ylim)
        plt.xlim(0, duration)
        plt.ylabel("neuron ID")

        if onset_times is not None:
            #plot block of translucent colour per pattern
            ax = plt.gca()
            pattern_legend=[]
            legend_labels=[]
            colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k','w']
            # labels = ['A','B','C',]
            for i,pattern in enumerate(onset_times):
                pattern_legend.append(
                    plt.Line2D([0], [0], color=colours[i%8], lw=4, alpha=0.2))
                legend_labels.append("s{}".format(i+1))
                for onset in pattern:
                    x_block = (onset, onset + scale_factor * pattern_duration)
                    ax.fill_between(
                        x_block, ylim, alpha=0.2, facecolor=colours[i%8],
                        lw=0.5)
            plt.legend(
                pattern_legend, legend_labels,
                bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=len(onset_times), mode="expand", borderaxespad=0.)
        if xlim is not None:
            plt.xlim(xlim)
        if legend_strings is not None and (
                subplots is None or subplots[2] == 1):
            plt.legend(
                legend_strings, bbox_to_anchor=(0.1, 1.25), loc='upper center',
                ncol=len(legend_strings),markerscale=10.)
        if filepath is not None:
            if subplots is None or subplots[2]==subplots[0]:
                plt.savefig(
                    filepath + '/' +
                    file_name + '{}.'.format(title) + file_format)

