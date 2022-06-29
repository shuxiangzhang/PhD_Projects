install.packages('MASS')
install.packages('dplyr')
install.packages('cramer')
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("GSAR")
install.packages('pryr')
install.packages("twosamples")# cvm_test
install.packages("DescTools")#RunsTest
library(DescTools)
library(twosamples)
library(pryr)
library(GSAR)
library(dplyr)
library(cramer)
library(MASS)

stream_names<-c('el_g.csv','el_s.csv','asc_g.csv','asc_s.csv','cyc_g.csv','cyc_s.csv','des_g.csv','des_s.csv','iro_g.csv','iro_s.csv','lpine_g.csv','lpine_s.csv','spruce_g.csv','spruce_s.csv','vac_g.csv','vac_s.csv')
for (stream_name in stream_names){
stream<-read.table(stream_name,stringsAsFactors = FALSE)
stream<-as.matrix(stream)
win1<-stream[1:100,]
win2<-stream[101:200,]
ave_1<-colMeans(win1)
f_1<-apply(win1,1,function(x) dist(rbind(x,ave_1)))
T_TIME<-NULL
MEM<-NULL        
for (j in 201:251){
  start<-Sys.time()
  print(j)
  # Break the loop if no drift was detected after the whole sliding window has entered the new distribution.
  win2<-win2[-1,]
  win2<-rbind(win2,stream[j,])
  ave_2<-colMeans(win2)
  f_2<-apply(win2,1,function(x) dist(rbind(x,ave_1)))
    result<-cvm_test(f_1,f_2,nboots = 1000)
    p_value<-unname(result[2])
    if (p_value<0.05){
      print(paste('Drift was detected at:',j))
    }
    end<-Sys.time()
    t_time<-end-start
    print(t_time)
    T_TIME<-c(T_TIME,as.numeric(t_time))
    mem<-object.size(win1)+object.size(win2)+object.size(result)
    MEM<-c(MEM,as.numeric(mem))
  }
  m_time<-mean(T_TIME)
  sd_time<-sd(T_TIME)
  m_mem<-mean(MEM)
  sd_mem<-sd(mem)
  print(m_mem)
  print(m_time)
  df<-data.frame(m_time=m_time,sd_time=sd_time,m_mem=m_mem,sd_mem=sd_mem)
write.csv(df,file = paste('uni_cramer_cost_',stream_name,'.csv',sep = ''))
}


