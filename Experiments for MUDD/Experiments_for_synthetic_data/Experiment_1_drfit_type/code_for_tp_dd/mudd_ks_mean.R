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

detect<-function(rho,sd,L,step_size,d){
  
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
    stream<-stream_generator(rho,sd,L,step_size,d)
    win1<-stream[1:100,]
    win2<-stream[101:200,]
    detect<-NULL
    flag<-TRUE
    counter<-0
    latest_point<-200
    for (j in 201:L){
      # Break the loop if no drift was detected after the whole sliding window has entered the new distribution.
      if ((j-(L-1000))>=200){
        break
      }
      win2<-win2[-1,]
      win2<-rbind(win2,stream[j,])
      if (flag==TRUE){
        if (j-latest_point == 1){
          print(j)
          latest_point<-j
          start_time = Sys.time()
          ave_1<-colMeans(win1)
          ave_2<-colMeans(win2)
          f_1<-apply(win1,1,function(x) dist(rbind(x,ave_1)))
          f_2<-apply(win2,1,function(x) dist(rbind(x,ave_1)))
          result<-ks.test(f_1,f_2)
          p_value<-result$p.value
          end_time = Sys.time()
          print(end_time-start_time)
          if (p_value<0.05){
            detect<-c(detect,j)
            print(paste('Drift was detected at:',j))
            if(j>=(L-1000) && j<=L){
              signal<-TRUE
              break
            }
            flag<-FALSE
            counter<-0
          }
        }
      }else{
        counter<-counter+1
        if (counter==100){
          win1<-win2
          flag<-TRUE
          latest_point<-latest_point+counter
        }
      }
    }
    if (signal==TRUE){
      TP<-TP+1
      dd<-j-(L-1000)
      DD<-c(DD,dd)
    }
  }
  
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
  write.csv(df,file = paste('uni_ks_mean_abrupt','_',d,'_',L,'_','.csv',sep = ''))
}


set.seed(0)
detect(rho = 0.2,sd = 0.2,L = 50000,step_size = 0.5,d = 2)


