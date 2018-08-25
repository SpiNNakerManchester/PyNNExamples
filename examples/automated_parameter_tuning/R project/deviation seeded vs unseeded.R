library(ggplot2)

ggplot(aes(x=as.numeric(row.names(seeded))-1), data=seeded) +
  
  
  #scale_x_continuous(limits = c(0,121), breaks=seq(0,120,1), expand = c(0, 0)) +
  scale_y_continuous(labels = scales::percent) +
  geom_point(aes(y=as.vector(seeded["std"][,1]), colour = "seeded")) +
  geom_smooth(aes(y=as.vector(seeded["std"][,1]), colour="seeded"), formula=y~x, method ='lm', se = FALSE) +
  geom_point(aes(y=as.vector(unseeded["std"][,1]), colour = "unseeded")) +
  geom_smooth(aes(y=as.vector(unseeded["std"][,1]), colour= "unseeded"), formula=y~x, method ='lm', se = FALSE) +
  xlab("Generation") + ylab("Training Accuracy Standard Deviation") +
  theme_bw() +
  theme(legend.position=c(.1, .2)) +
  theme(legend.title = element_blank()) +
  theme(legend.background = element_rect(size=0.5, linetype="solid", colour ="black"))+
  theme(panel.grid.minor = element_blank())


