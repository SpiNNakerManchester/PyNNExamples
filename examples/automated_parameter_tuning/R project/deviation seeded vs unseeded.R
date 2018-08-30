pearsons <- function(arg1, arg2){
  return(cor.test(arg1, arg2, method = "pearson", conf.level = 0.95))
}



library(ggplot2)

datas <- rbind(cbind(generation=1:nrow(seeded), seeded, seeded=c("seeded")),cbind(generation=1:nrow(unseeded),unseeded, seeded=c("unseeded")))
ggplot(datas[which(datas$generation<228),], aes(x=generation, y=std, group=seeded)) +
  scale_y_continuous(labels = scales::percent) +
  xlab("Generation") + ylab("Training Accuracy Standard Deviation") +
  geom_point(aes(colour=seeded, shape=seeded)) +
  geom_smooth(aes(colour=seeded), method='lm', se=FALSE) +
  theme(legend.title = element_blank()) +
  theme_bw() +
  theme(legend.position=c(.1, .2)) +
  theme(legend.title = element_blank()) +
  theme(legend.background = element_rect(size=0.5, linetype="solid", colour ="black"))+
  theme(panel.grid.minor = element_blank())

