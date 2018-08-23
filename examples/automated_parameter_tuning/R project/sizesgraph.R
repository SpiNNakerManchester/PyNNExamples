library(ggplot2)

ggplot(data=network_sizes, aes(Neurons, Synapses, label=paste(Citation, Year))) +
  geom_point(na.rm = TRUE) +
  geom_text(aes(label=paste(Citation, Year, sep = ", ")),hjust=0, vjust=-0.1, check_overlap = F, na.rm = TRUE) +
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x)),
    limits = c(1E4, 1E10)
  ) +
  scale_y_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x)),
    limits = c(1E4, 1E12)
  ) +
  xlab('Number of Neurons') +
  ylab('Number of Synapses') +
  theme_bw()

