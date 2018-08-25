library(ggplot2)

ggplot(aes(x=as.numeric(row.names(seeded))-1), data=seeded) +
  
  
  scale_x_continuous(limits = c(0,121), breaks=seq(0,120,1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0,1), breaks=seq(0,1,0.1), expand = c(0, 0), labels = scales::percent) +
  scale_colour_hue(l=70)  +
  geom_line(aes(y=as.vector(seeded["avg"][,1]), colour = "seeded")) +
  #geom_smooth(aes(y=as.vector(seeded["avg"][,1])), formula=y~x, method ='lm') +
  geom_line(aes(y=as.vector(unseeded["avg"][,1]), colour = "unseeded")) +
  #geom_smooth(aes(y=as.vector(unseeded["avg"][1:116,1])), formula=y~x, method ='lm') +
  xlab("Generation") + ylab("Training Accuracy") +
  theme_bw() +
  theme(legend.position=c(.1, .8)) +
  theme(legend.title = element_blank()) +
  theme(legend.background = element_rect(size=0.5, linetype="solid", colour ="black"))+
  theme(panel.grid.minor = element_blank())
  


