library(ggplot2)

ggplot(aes(x=as.numeric(row.names(checkpoint.pkl))-1), data=checkpoint.pkl) +
  scale_x_continuous(limits = c(0,121), breaks=seq(0,120,10), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0,1), breaks=seq(0,1,0.1), expand = c(0, 0), labels = scales::percent) +
  scale_colour_hue(l=70)  +
  geom_ribbon(aes(ymin = avg - std, ymax = avg + std), fill = "grey90") +
  geom_line(aes(y = avg, colour = "Mean")) +
  geom_line(aes(y = min, colour = "Minimum")) +
  geom_line(aes(y = max, colour = "Maximum")) +
  xlab("Generation") + ylab("Training Accuracy") +
  theme_bw() +
  theme(legend.position=c(.1, .8)) +
  theme(legend.title = element_blank()) +
  theme(legend.background = element_rect(size=0.5, linetype="solid", colour ="black"))+
  theme(panel.grid.minor = element_blank())
