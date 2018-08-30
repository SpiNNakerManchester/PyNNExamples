library(ggplot2)
ggplot(data, aes(x=generation, y=avg, group=seeded)) +
  scale_y_continuous(labels = scales::percent, limits=c(0,1)) +
  geom_point(aes(colour=seeded)) +
  geom_point(aes(y=max, colour=seeded)) +
  geom_point(aes(y=min, colour=seeded)) +
  theme_bw() +
  theme(legend.position=c(.1, .8)) +
  theme(legend.title = element_blank()) +
  theme(legend.background = element_rect(size=0.5, linetype="solid", colour ="black"))+
  theme(panel.grid.minor = element_blank())
  