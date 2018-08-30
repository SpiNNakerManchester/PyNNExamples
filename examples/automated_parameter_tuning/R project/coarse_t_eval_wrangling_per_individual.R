library(ggplot2)
require(reshape2)
library(forcats)
library(dplyr)
library(readr)
coarse_data <- read_csv("timing/coarse data.csv", 
                        col_names = FALSE)

titles <- c("Subpopulation size", "Population size", "number of processes", "t_start_gen", "t_end_select", "t_end_variation", "t_end_preprocessing", "t_end_evaluatepop", "t_end_postprocessing", 
            "t_end_stats", "t_end_gen", "sevens", "t_min","t_setup","t_run",
            "t_gather","t_cost","avg_retry")
colnames(coarse.data)<- titles

head(coarse.data)


#selected <- coarse.data[which((coarse.data$`Subpopulation size`) %% 20 ==0),]
selected <- coarse.data
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
head(selected)
selected <-aggregate(selected, list(selected$`Subpopulation size`), mean)
colnames(selected)[17] <- "t_fitness"
wanted <- c("Subpopulation size", "t_start", "t_setup", "t_run", "t_gather", "t_fitness")
selected<- selected[wanted]
selected<-transmute(selected, `Subpopulation size`=`Subpopulation size`,
          t_start = t_start/`Subpopulation size`,
          t_setup = t_setup/`Subpopulation size`,
          t_run = t_run/`Subpopulation size`,
          t_gather = t_gather/`Subpopulation size`,
          t_fitness= t_fitness/`Subpopulation size`)
head(selected)
#selected<-selected[which(selected$`Subpopulation size`<520),]
selected<-melt(selected, id.vars="Subpopulation size")
ggplot(selected[order(selected$variable), ], aes(x = `Subpopulation size`, y = value, fill = variable)) + 
  geom_col(position = position_stack(reverse = TRUE)) +
  scale_fill_brewer(palette="Set1") + 
  theme_bw() +
  theme(legend.direction = "horizontal") +
  theme(legend.title = element_blank()) +
  theme(legend.background = element_rect(size=0.5, linetype="solid", colour ="black"))+
  #theme(legend.text=element_text(size=7)) +
  theme(legend.direction="vertical")+
  theme(legend.position=c(0.8,0.95), legend.justification=c(0, 1))+
  labs(x = "Subpopulation size", y="Average execution time per individual/s")
