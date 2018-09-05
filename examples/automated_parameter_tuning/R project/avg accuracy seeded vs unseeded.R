library(ggplot2)
library(ggrepel)

datas <- rbind(cbind(generation=1:nrow(seeded), seeded, seeded=c("seeded")),cbind(generation=1:nrow(unseeded),unseeded, seeded=c("unseeded")))

ggplot(datas[which(datas$generation<305),], aes(x=generation, y=avg, group=seeded)) +
  scale_x_continuous(breaks=seq(0,305,10)) + 
  scale_y_continuous(labels = scales::percent, limits=c(0,1), breaks=seq(0,1,0.1)) +
  geom_line(aes(colour=seeded)) +
  geom_line(aes(y=max, colour=seeded)) +
  geom_line(aes(y=min, colour=seeded)) +
  geom_label(x=-1, y=0.5, label="Maximum") +
  geom_label(x=-1, y=0.23, label="Mean") +
  geom_label(x=-1, y=0.075, label="Minimum") +
  theme_bw() +
  theme(legend.position=c(.1, .8)) +
  xlab("Generation") + ylab("Training Accuracy") +
  theme(legend.position=c(.1, .8)) +
  theme(legend.title = element_blank()) +
  theme(legend.background = element_rect(size=0.5, linetype="solid", colour ="black"))+
  theme(panel.grid.minor = element_blank())
  