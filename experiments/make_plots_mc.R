options(width=160)

library(tidyverse)
library(kableExtra)

setwd("~/GitHub/Conformalized_early_stopping")
idir <- "tmp_results/mc/result1/multiclass/"
ifile.list <- list.files(idir)

##tab.dir <- "tables"
tab.dir <- "tmp_results/tables"
fig.dir <- "tmp_results/figures"

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

key.values <- c("M-coverage", "M-size")
key.labels <- c("Coverage (marginal)", "Cardinality")

Method.values <- c("CES", "Naive", "Theory", "Data Splitting", "Full Training")
Method.labels <- c("CES", "Naive", "Naive + theory", "Data splitting", "Full training")

color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange", "red")
shape.scale <- c(15, 4, 8, 1, 5)


results_mc <- results.raw %>%
  mutate(Method = factor(Method, Method.values, Method.labels)) %>%
  pivot_longer(cols=c(`M-coverage`, `M-size`), names_to='Key', values_to='Value') %>%
  mutate(Key = factor(Key, key.values, key.labels)) %>%
  group_by(Method, batch_size, lr, n_data, n_epoch, Key) %>%
  summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T)) %>%
  filter(Method!='Naive', lr==0.01, n_epoch==50, n_data!=200)


## Make nice plots for paper
make_plot <- function(xmax=2000) {
  plot.alpha <- 0.1
  df.nominal <- tibble(Key=c("M-coverage"), Value=1-plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  df.ghost <- tibble(Key=c("M-coverage","M-coverage"), Value=c(0.7,1), Method="CES") %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  pp <- results_mc %>%
    filter(Method != "NA") %>%
    ggplot(aes(x=n_data, y=Value, color=Method, shape=Method)) +
    geom_point(alpha=0.75) +
    geom_line() +
    ##    geom_errorbar(aes(ymin=Value.low, ymax=Value.upp)) +
    geom_hline(data=df.nominal, aes(yintercept=Value)) +
    geom_point(data=df.ghost, aes(x=100,y=Value), alpha=0) +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    facet_wrap(.~Key, scales="free") +
    scale_x_continuous(trans='log10', lim=c(500,xmax), breaks=c(500,1000,2000)) +
    #        scale_y_continuous(trans='log10') +
    xlab("Sample size") +
    ylab("") +
    theme_bw()
  ggsave(sprintf("%s/exp_mc.pdf", fig.dir), pp, device=NULL, width=5.5, height=2)
}

make_plot()


## Make nice tables for paper
make_table <- function(xmax=2000) { 
  plot.alpha <- 0.1
  df <- results_mc %>%
    filter(!is.na(Method)) %>%
    mutate(Value.str = sprintf("%.3f (%.3f)", Value, Value.se)) %>%
    mutate(Value.str=ifelse((Key=="Coverage (marginal)")*(Value<0.85)==1, sprintf("\\textcolor{red}{%s}", Value.str), Value.str)) %>%
    group_by(n_data, Key) %>%
    mutate(Value.best = min(Value)) %>%
    ungroup() %>%
    mutate(Value.str=ifelse((Key=="Cardinality")*(Value-Value.se<Value.best)==1, sprintf("\\textbf{%s}", Value.str), Value.str)) %>%
    select(Method, n_data, Key, Value.str) %>%
    pivot_wider(names_from="Key", values_from="Value.str") %>%
    arrange(n_data, Method) %>%
    select(n_data, Method, Cardinality, everything())
  tb1 <- df %>%
    kbl("latex", booktabs=TRUE, longtable = TRUE, escape = FALSE, caption = NA,
        col.names = c("Sample size", "Method", "Cardinality", "Marignal coverage")) %>%
    pack_rows(index = table(df$n_data))
  
  
  tb1 %>% save_kable(sprintf("%s/exp_mc.tex", tab.dir), keep_tex=TRUE, self_contained=FALSE)
}

make_table()