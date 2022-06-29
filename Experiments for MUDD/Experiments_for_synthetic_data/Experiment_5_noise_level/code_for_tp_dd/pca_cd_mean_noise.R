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

stream_generator<-function(rho,sd,L,step_size,d){
  mu_1<-0.2
  mu<-rep(mu_1,d)
  Sigma<-matrix(nrow = d,ncol = d)
  for(i in seq(d)){
    for(j in seq(d)){
      if (i==j){
        Sigma[i,j]<-sd^2
      }else{
        Sigma[i,j]<-sd^2*rho
      }
    }
  }
  s_1<-mvrnorm((L-1000), mu, Sigma )
  mu_1<-mu_1+step_size
  mu<-rep(mu_1,d)
  print(mu)
  s_2<-mvrnorm(1000, mu, Sigma )
  stream<-rbind(s_1,s_2)
  return(stream)
}


noise_masking<-function(stream,level,mu,sd,d){
  low<-mu-3*sd
  up<-mu+3*sd
  data_point<-runif(d,low,up)
  for (row in nrow(stream)){
    num<-runif(1,0,1)
    if (num<=level){
      stream[row,]<-data_point
    }
  }
  return (stream)
}

detect<-function(rho,sd,L,step_size,d,level){
  
  delta<-0.005
  alpha<-1-0.0001
  TP<-0
  DD<-NULL
  D_time<-NULL
  MEM<-NULL
  T_time<-NULL
  T_time_num<-NULL
  D_location<-list()
  for (i in seq(30)){
    dd<-NULL
    signal<-FALSE
    kl_count<-0
    kl_mean<-0
    dif<-0
    stream<-stream_generator(rho,sd,L,step_size,d)
    stream<-noise_masking(stream,level,0.2,sd,d)
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

    
    for (n in 201:L){
      print(n)
      start_time<-Sys.time()
      # Break the loop if no drift was detected after the whole sliding window has entered the new distribution.
      if ((n-(L-1000))>=200){
        break
      }
      kl_count<-kl_count+1
      win2<-win2[-1,]
      win2<-rbind(win2,stream[n,])
      
      new_win2<-as.data.frame(scale(win2, pca$center, pca$scale) %*% pca$rotation)
      new_win2<-new_win2[,1:k]
      
      kl_x_y<-rep(0,k)
      kl_y_x<-rep(0,k)
      kl<-rep(0,k)
      
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
      kl_mean <- kl_mean + (kl - kl_mean) / kl_count
      dif<- max(0,alpha * dif + (kl - kl_mean - delta))
      #print(paste('kl=',kl))
      #print(paste('kl_mean=',kl_mean))
      threshold<-0.05*100*kl_mean
      if (n%%100==0){
        m<-100
      }else{
        m<-n%%100
      }
      end_time<-Sys.time()
      print(end_time-start_time)
      #print(paste('dif=',dif))
      #print(threshold)
      if (dif>=threshold){
        print(paste('Drift was detected at',n))
      
            if(n>=(L-1000) && n<=L){
              TP<-TP+1
              dd<-n-(L-1000)
              DD<-c(DD,dd)
              break
            }

      }}}

  if (length(DD)==0){
    m_dd<-0
    sd_dd<-0
  }else if(length(DD)==1){
    m_dd<-mean(DD)
    sd_dd<-0
  }else{
    m_dd<-mean(DD)
    sd_dd<-sd(DD)
  }
  
  m_dd<-format(round(m_dd, 2), nsmall = 2)
  sd_dd<-format(round(sd_dd, 2), nsmall = 2)
  
  df<-data.frame(Detection_Delay=paste(m_dd,'/',sd_dd),tp=TP)
  write.csv(df,file = paste('PCA-CD_mean_abrupt_noise','_',d,'_',L,'_',level,'.csv',sep = ''))
}


set.seed(0)
for (l in seq(0.05,0.3,0.05)){
  detect(rho = 0.2,sd = 0.2,L = 50000,step_size = 0.5,d = 2, level = l)
}
