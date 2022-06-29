install.packages('MASS')
install.packages('dplyr')
install.packages('cramer')
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("GSAR")
install.packages('pryr')
library(pryr)
library(GSAR)
library(dplyr)
library(cramer)
library(MASS)

stream_generator<-function(rho,sd,L){
  mu_1<-0.2
  mu_2<-0.2
  mu<-c(mu_1,mu_2)
  Sigma<-matrix(c(sd^2, sd^2*rho, sd^2*rho, sd^2),2)
  stream<-mvrnorm(L, mu, Sigma )
  return(stream)
}

run<-function(rho,sd,L){
  
  delta<-0.005
  alpha<-1-0.0001
  fp<-NULL
  MEM<-NULL
  T_time<-NULL
  d<-2
  for (w in seq(30)){
    print('next')
    d_time<-NULL
    mem<-NULL
    kl_count<-0
    kl_mean<-0
    dif<-0
    stream<-stream_generator(rho,sd,L)
    win1<-stream[1:100,]
    win2<-stream[101:200,]
    pca <- prcomp(win1, center = TRUE,scale. = TRUE)
    # Determine the number (k) of PCS to use.
    eigs <- pca$sdev^2
    prop_of_var<-eigs / sum(eigs)
    cum_prop<-cumsum(prop_of_var)
    k<-which(cum_prop>=0.999)[1]
    
    # Project data in win1 and win2 to PC space
    new_win1<-as.data.frame(scale(win1, pca$center, pca$scale) %*% pca$rotation)
    new_win1<-new_win1[,1:k]
    
    detect<-NULL
    latest_point<-200
    for (u in 201:nrow(stream)){
      kl_x_y<-rep(0,k)
      kl_y_x<-rep(0,k)
      kl<-rep(0,k)
      start_time = Sys.time()
      win2<-win2[-1,]
      win2<-rbind(win2,stream[u,])
      if (u-latest_point == 100){
        latest_point<-u
        kl_count<-kl_count+1
        new_win2<-as.data.frame(scale(win2, pca$center, pca$scale) %*% pca$rotation)
        new_win2<-new_win2[,1:k]
        
        for (i in 1:k){
          h_1<-hist(new_win1[,i],breaks = 100/10)
          h_2<-hist(new_win2[,i],breaks = 100/10)
          
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
        #print(kl)
        kl_mean <- kl_mean + (kl - kl_mean) / kl_count
        dif<- max(0,alpha * dif + (kl - kl_mean - delta))
        #print(paste('kl=',kl))
        #print(paste('kl_mean=',kl_mean))
        threshold<-0.05*100*kl_mean
        print(paste('threshold = ',threshold))
        if (n%%100==0){
          m<-100
        }else{
          m<-n%%100
        }
        print(paste('dif=',dif))
        if (dif>=threshold){
          print(paste('Drift was detected at',u))
          detect<-c(detect,j)
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
          dif<-0
        }
        end_time = Sys.time()
        t <-end_time-start_time
        #print(t)
        d_time<-c(d_time,as.numeric(t))
      }
    }
    fp<-c(fp,length(detect))
    T_time<-c(T_time,mean(d_time)) # Record the total execution time when no drift is detected.
    print(T_time)
    mem<-object.size(win1)+object.size(win2)+object.size(pca)+object.size(h_1)+object.size(h_2)
    MEM<-c(MEM,as.numeric(mem))
  }
  
  m_fpr<-mean(fp/L)
  sd_fpr<-sd(fp/L)
  
  m_execution_time<-mean(T_time)
  sd_execution_time<-sd(T_time)
  
  m_memory<-mean(MEM)
  sd_memory<-sd(MEM)
  
  df<-data.frame(fpr=paste(m_fpr,'/',sd_fpr),Memory_usage=paste(m_memory,'/',sd_memory),execution_time=paste(m_execution_time,'/',sd_execution_time))
  write.csv(df,file = paste('pca_cd_len_fp_',L,'.csv',sep = ''))
}

set.seed(0)
for (l in (c(seq(20000,100000,20000),500000,1000000))){
  run(rho = 0.2,sd = 0.2,L = l)
}