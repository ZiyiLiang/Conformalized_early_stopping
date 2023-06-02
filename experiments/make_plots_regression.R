options(width=160)

library(tidyverse)
library(kableExtra)

idir <- "results_hpc/exp1/"
ifile.list <- list.files(idir)

##tab.dir <- "tables"
tab.dir <- "/home/msesia/Workspace/research/active/conformalized-es/paper/tables_new"
fig.dir <- "figures"

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
    df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

key.values <- c("marg_coverage", "cond_coverage", "avg_size")
key.labels <- c("Coverage (marginal)", "Coverage (conditional)", "Width")

method.values <- c("ces", "naive", "theory", "benchmark", "naive-full")
method.labels <- c("CES", "Naive", "Naive + theory", "Data splitting", "Full training")

color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange", "red")
shape.scale <- c(15, 4, 8, 1, 5)


results <- results.raw %>%
    mutate(Method = factor(method, method.values, method.labels)) %>%
    pivot_longer(cols=c('marg_coverage','cond_coverage','avg_size'), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(data, Method, n, n_features, n_test, noise, lr, wd, alpha, alpha_2, optimizer, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T)) %>%
    mutate(Value.low = Value - 2*Value.se, Value.upp = Value + 2*Value.se) %>%
    mutate(Value.upp = ifelse((Key=="Coverage (conditional)")*(Value.upp>1)==1, 1, Value.upp)) %>%
    mutate(Value.low = ifelse((Key=="Coverage (conditional)")*(Value.low<0)==1, 0, Value.low)) %>%
    mutate(Value.upp = ifelse((Key=="Coverage (marginal)")*(Value.upp>1)==1, 1, Value.upp)) %>%
    mutate(Value.low = ifelse((Key=="Coverage (marginal)")*(Value.low<0)==1, 0, Value.low))


## Make nice plots for paper
make_plot <- function(plot.data, xmax=2000) {
    plot.alpha <- 0.1
    plot.noise <- 0.01
    plot.lr <- 0.001
    plot.wd <- 0
    df.nominal <- tibble(Key=c("marg_coverage","cond_coverage"), Value=1-plot.alpha) %>%
        mutate(Key = factor(Key, key.values, key.labels))    
    df.ghost <- tibble(Key=c("marg_coverage","cond_coverage","marg_coverage","cond_coverage"), Value=c(0.7,0.7,1,1), method="ces") %>%
        mutate(Method = factor(method, method.values, method.labels)) %>%
        mutate(Key = factor(Key, key.values, key.labels))    
    pp <- results %>%
        filter(Method!="Naive") %>%
        filter(Method != "NA") %>%
        filter(noise==plot.noise, data==plot.data, alpha==plot.alpha, lr==plot.lr, wd==plot.wd) %>%
        ggplot(aes(x=n, y=Value, color=Method, shape=Method)) +
        geom_point(alpha=0.75) +
        geom_line() +
        ##    geom_errorbar(aes(ymin=Value.low, ymax=Value.upp)) +
        geom_hline(data=df.nominal, aes(yintercept=Value), linetype=2) +
        geom_point(data=df.ghost, aes(x=100,y=Value), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        facet_wrap(.~Key, scales="free") +
        scale_x_continuous(trans='log10', lim=c(200,xmax), breaks=c(200,500,1000,2000)) +
#        scale_y_continuous(trans='log10') +
        xlab("Sample size") +
        ylab("") +
        theme_bw()
    ggsave(sprintf("%s/exp_regression_%s.pdf", fig.dir, plot.data), pp, width=7.5, height=2)
}
make_plot("bio")
make_plot("bike")
make_plot("concrete", xmax=1000)

## Make nice plots for paper
make_plot_small <- function(plot.data, xmax=2000) {
    plot.alpha <- 0.1
    plot.noise <- 0.01
    plot.lr <- 0.001
    plot.wd <- 0
    df.nominal <- tibble(Key=c("marg_coverage","cond_coverage"), Value=1-plot.alpha) %>%
        mutate(Key = factor(Key, key.values, key.labels))    
    df.ghost <- tibble(Key=c("marg_coverage","cond_coverage","marg_coverage","cond_coverage"), Value=c(0.7,0.7,1,1), method="ces") %>%
        mutate(Method = factor(method, method.values, method.labels)) %>%
        mutate(Key = factor(Key, key.values, key.labels))    
    pp <- results %>%
        filter(Method!="Naive") %>%
        filter(Method != "NA") %>%
        filter(noise==plot.noise, data==plot.data, alpha==plot.alpha, lr==plot.lr, wd==plot.wd) %>%
        ggplot(aes(x=n, y=Value, color=Method, shape=Method)) +
        geom_point(alpha=0.75) +
        geom_line() +
        ##    geom_errorbar(aes(ymin=Value.low, ymax=Value.upp)) +
        geom_hline(data=df.nominal, aes(yintercept=Value), linetype=2) +
        geom_point(data=df.ghost, aes(x=100,y=Value), alpha=0) +
        scale_color_manual(values=color.scale) +
        scale_shape_manual(values=shape.scale) +
        facet_wrap(.~Key, scales="free") +
        scale_x_continuous(trans='log10', lim=c(200,xmax), breaks=c(200,500,1000,2000)) +
                                        #        scale_y_continuous(trans='log10') +
        xlab("Sample size") +
        ylab("") +
        theme_bw() +
        theme(legend.position="bottom", legend.margin=margin(l=-0.1, t = -0.1, unit='cm'), plot.margin=grid::unit(c(0,0.25,0,-0.25), "cm"))
    ggsave(sprintf("%s/exp_regression_%s_small.pdf", fig.dir, plot.data), pp, width=5, height=2.25)
}
make_plot_small("bio")



## Make nice tables for paper
make_table <- function(plot.data, xmax=2000) { 
   plot.alpha <- 0.1
    plot.noise <- 0.01
    plot.lr <- 0.001
    plot.wd <- 0
    df <- results %>%
        filter(!is.na(Method)) %>%
        filter(data==plot.data, noise==plot.noise, alpha==plot.alpha, lr==plot.lr, wd==plot.wd) %>%
        mutate(Value.str = sprintf("%.3f (%.3f)", Value, Value.se)) %>%
        mutate(Value.str=ifelse((Key=="Coverage (marginal)")*(Value<0.85)==1, sprintf("\\textcolor{red}{%s}", Value.str), Value.str)) %>%
        mutate(Value.str=ifelse((Key=="Coverage (conditional)")*(Value<0.85)==1, sprintf("\\textcolor{red}{%s}", Value.str), Value.str)) %>%
        mutate(Data=data) %>%
        group_by(Data, n, Key) %>%
        mutate(Value.best = min(Value)) %>%
        ungroup() %>%
        mutate(Value.str=ifelse((Key=="Width")*(Value-Value.se<Value.best)==1, sprintf("\\textbf{%s}", Value.str), Value.str)) %>%
        select(Data, Method, n, Key, Value.str) %>%
        pivot_wider(names_from="Key", values_from="Value.str") %>%
        arrange(n, Method) %>%
        select(n, Data, Method, Width, everything())
    tb1 <- df %>%
        kbl("latex", booktabs=TRUE, longtable = FALSE, escape = FALSE, caption = NA,
                    col.names = c("Sample size", "Data", "Method", "Width", "Marginal", "Conditional")) %>%
        add_header_above(c(" " = 4, "Coverage" = 2)) %>%
        pack_rows(index = table(df$n))        
    tb1 %>% save_kable(sprintf("%s/exp_regression_%s.tex", tab.dir, plot.data), keep_tex=TRUE, self_contained=FALSE)
}
make_table("bio")
make_table("bike")
make_table("concrete")

## if(FALSE) {
##     plot.data <- "friedman1"
##     plot.n <- 1000
##     plot.alpha <- 0.1
##     plot.lr <- 0.001
##     plot.wd <- 0
##     pp <- results %>%
##         filter(Key=="Coverage (marginal)", data==plot.data, n_train==plot.n, alpha==plot.alpha, lr==plot.lr, wd==plot.wd) %>%
##         ggplot(aes(x=n_cal, y=Value, color=Method)) +
##         geom_point() +
##         geom_line() +
##         geom_errorbar(aes(ymin=Value-2*Value.se, ymax=Value+2*Value.se)) +
##         geom_hline(yintercept=1-plot.alpha) +
##         facet_grid(n_features~noise) +
##         scale_x_continuous(trans='log10') +
##         theme_bw()
##     pp
## }


## if(TRUE) {
##     plot.data <- "friedman1"
##     plot.n <- 1000
##     plot.alpha <- 0.1
##     plot.noise <- 0.01
##     plot.lr <- 0.001
##     plot.wd <- 0
##     df.nominal <- tibble(Key=c("marg_coverage","cond_coverage"), Value=1-plot.alpha) %>%
##         mutate(Key = factor(Key, key.values, key.labels))    
##     df.ghost <- tibble(Key=c("marg_coverage","cond_coverage","marg_coverage","cond_coverage"), Value=c(0.7,0.7,1.2,1.2), Method=NA) %>%
##         mutate(Key = factor(Key, key.values, key.labels))    
##     pp <- results %>%
##         #filter(noise==plot.noise, data==plot.data, n_train==plot.n, alpha==plot.alpha, lr==plot.lr, wd==plot.wd) %>%
##         ggplot(aes(x=n, y=Value, color=Method, shape=Method)) +
##         geom_point(alpha=0.5) +
##         geom_line() +
##         geom_errorbar(aes(ymin=Value-2*Value.se, ymax=Value+2*Value.se)) +
##         geom_hline(data=df.nominal, aes(yintercept=Value)) +
##         geom_point(data=df.ghost, aes(x=100,y=Value), alpha=0) +
##         facet_wrap(data~Key, scales="free") +
##         scale_x_continuous(trans='log10') +
##         scale_y_continuous(trans='sqrt') +
##         xlab("Number of data points") +
##         ylab("") +
##         theme_bw()
##     pp
## }

## ggsave("plot_ces.pdf", pp)


