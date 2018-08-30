library(ggplot2)
require(reshape2)
library(forcats)
library(dplyr)
titles <- c("Subpopulation size", "Population size", "number of processes", "t_start_gen", "t_end_select", "t_end_variation", "t_end_preprocessing", "t_end_evaluatepop", "t_end_postprocessing", 
            "t_end_stats", "t_end_gen", "sevens", "t_min","t_setup","t_run",
            "t_gather","t_cost","avg_retry")
colnames(fine.data)<- titles

selected <- fine.data

cleandata <- function(x){
  mutate(x, t_select = t_end_select - t_start_gen) %>%
    mutate(t_variation = t_end_variation - t_end_select) %>%
    mutate(t_preprocessing = t_end_preprocessing - t_end_variation) %>%
    mutate(t_evaluation=t_end_evaluatepop - t_end_preprocessing) %>%
    mutate(t_postprocessing= t_end_postprocessing - t_end_evaluatepop) %>%
    mutate(t_statistics=t_end_stats-t_end_postprocessing) %>%
    mutate(t_save = t_end_gen-t_end_stats) %>%
    mutate(t_gen = t_end_gen - t_start_gen) %>%
    mutate(t_start = t_min - t_end_preprocessing) %>%
    select(-sevens) %>%
    select(-t_gen)
}
selected<-cleandata(selected)
selected <-aggregate(selected, list(selected$`Subpopulation size`), mean)
colnames(selected)[17]
colnames(selected)[17] <- "t_fitness"
selected<- cbind(selected$`Subpopulation size`,selected[-1]/selected$`Subpopulation size`)
colnames(selected)[1]<-"Subpopulation size"
selected <-aggregate(selected, list(selected$`Subpopulation size`), mean)

wanted <- c("Subpopulation size", "t_start", "t_setup", "t_run", "t_gather", "t_fitness")
head(selected)
selected<- selected[wanted]
selected<-melt(selected, id.vars="Subpopulation size")
ggplot(selected[order(selected$variable), ], aes(x = `Subpopulation size`, y = value, fill = variable)) + 
  geom_col(position = position_stack(reverse = TRUE)) +
  scale_fill_brewer(palette="Set1") + 
  theme_bw() +
  theme(legend.title = element_blank()) +
  theme(legend.background = element_rect(size=0.5, linetype="solid", colour ="black"))+
  #theme(legend.text=element_text(size=7)) +
  theme(legend.direction="horizontal")+
  theme(legend.position=c(0.55,0.95), legend.justification=c(0, 1))+
  labs(x = "Subpopulation size", y="Average execution time per individual/s")

