library(pryr)
library(GSAR)
library(dplyr)
library(cramer)
library(MASS)

stream_names<-c('el_g.csv','el_s.csv','asc_g.csv','asc_s.csv','cyc_g.csv','cyc_s.csv','des_g.csv','des_s.csv','iro_g.csv','iro_s.csv','lpine_g.csv','lpine_s.csv','spruce_g.csv','spruce_s.csv','vac_g.csv','vac_s.csv')
for(stream_name in stream_names){stream<-read.table(stream_name,stringsAsFactors = FALSE)
stream<-as.matrix(stream)
win1<-stream[1:100,]
win2<-stream[101:200,]
detect<-NULL
delta<-0.005
alpha<-1-0.0001
kl_count<-0
kl_mean<-0
dif<-0
T_TIME<-NULL
MEM<-NULL
pca <- prcomp(win1, center = TRUE,scale. = TRUE)
# Determine the number (k) of PCS to use.
eigs <- pca$sdev^2
prop_of_var<-eigs / sum(eigs)
cum_prop<-cumsum(prop_of_var)
k<-which(cum_prop>=0.999)[1]

# Project data in win1 and win2 to PC space
new_win1<-as.data.frame(scale(win1, pca$center, pca$scale) %*% pca$rotation)
new_win1<-new_win1[,1:k]

for (data_point in 201:251){
  print(data_point)
  start<-Sys.time()
  kl_count<-kl_count+1
  win2<-win2[-1,]
  win2<-rbind(win2,stream[data_point,])
  
  new_win2<-as.data.frame(scale(win2, pca$center, pca$scale) %*% pca$rotation)
  new_win2<-new_win2[,1:k]
  
  kl_x_y<-rep(0,k)
  kl_y_x<-rep(0,k)
  kl<-rep(0,k)
  
  for (i in 1:k){
    h_1<-hist(new_win1[,i],breaks = 100/10,plot=FALSE)
    h_2<-hist(new_win2[,i],breaks = 100/10,plot=FALSE)
    
    for(j in 1:100){
      x<-new_win1[,i][j]
      y<-new_win2[,i][j]
      interval_no_x<-(which(h_1$breaks>=x)[1]-1)
      interval_no_y<-(which(h_2$breaks>=y)[1]-1)
      prob_x<-h_1$density[interval_no_x]
      prob_y<-h_2$density[interval_no_y]
      kl_x_y[i]<-kl_x_y[i]+prob_x*log(prob_x/prob_y)
      kl_y_x[i]<-kl_y_x[i]+prob_y*log(prob_y/prob_x)
    }
  }
  kl_merge<-rbind(kl_x_y,kl_y_x)
  kl<-apply(kl_merge,2,max)
  kl<-max(kl)
  kl_mean <- kl_mean + (kl - kl_mean) / kl_count
  dif<- max(0,alpha * dif + (kl - kl_mean - delta))
  #print(paste('kl=',kl))
  #print(paste('kl_mean=',kl_mean))
  threshold<-0.05*100*kl_mean
  #print(paste('dif=',dif))
  #print(threshold)
  if (dif>=threshold){
    detect<-c(detect,data_point)
    print(paste('Drift was detected at:',data_point))
    win1<-win2
    pca <- prcomp(win1, center = TRUE,scale. = TRUE)
    # Determine the number (k) of PCS to use.
    eigs <- pca$sdev^2
    prop_of_var<-eigs / sum(eigs)
    cum_prop<-cumsum(prop_of_var)
    k<-which(cum_prop>=0.999)[1]
    
    # Project data in win1 and win2 to PC space
    new_win1<-as.data.frame(scale(win1, pca$center, pca$scale) %*% pca$rotation)
    new_win1<-new_win1[,1:k]
  }
  end<-Sys.time()
  t_time<-end-start
  print(t_time)
  T_TIME<-c(T_TIME,as.numeric(t_time))
  mem<-object.size(win1)+object.size(win2)+object.size(pca)+object.size(h_1)+object.size(h_2)
  MEM<-c(MEM,as.numeric(mem))
}
m_time<-mean(T_TIME)
sd_time<-sd(T_TIME)
m_mem<-mean(MEM)
sd_mem<-sd(mem)
print(m_mem)
print(m_time)
df<-data.frame(m_time=m_time,sd_time=sd_time,m_mem=m_mem,sd_mem=sd_mem)

write.csv(df,file = paste('pca_cost_',stream_name,'.csv',sep = ''))
}
