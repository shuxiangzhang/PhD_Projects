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

stream_generator<-function(rho,sd,L,d){
  mu_1 <-0.2
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

  stream<-mvrnorm(L, mu, Sigma )
  return(stream)
}

detect<-function(rho,sd,L,d,win_size){
  fp<-NULL
  MEM<-NULL
  T_time<-NULL
  for (i in seq(30)){
    d_time<-NULL
    mem<-NULL
    stream<-stream_generator(rho,sd,L,d)
    win1<-stream[1:win_size,]
    win2<-stream[(win_size+1):2*win_size,]
    detect<-NULL
    flag<-TRUE
    counter<-0
    latest_point<-2*win_size
    for (j in (2*win_size+1):nrow(stream)){
      print(j)
      start_time = Sys.time()
      win2<-win2[-1,]
      win2<-rbind(win2,stream[j,])
      win<-rbind(win1,win2)
      if (flag==TRUE){
        if (j-latest_point == win_size){
          latest_point<-j
          result<-WWtest(object=t(win), group=c(rep(1,win_size),rep(2,win_size)))
          p_value<-result
          if (p_value<0.05){
            detect<-c(detect,j)
            print(paste('Drift was detected at:',j))
            flag<-FALSE
            counter<-0
          }
          end_time = Sys.time()
          t <-end_time-start_time
          print(t)
          d_time<-c(d_time,as.numeric(t))
        }
      }else{
        counter<-counter+1
        if (counter==win_size){
          win1<-win2
          flag<-TRUE
          latest_point<-latest_point+counter
        }
      }
    }
    fp<-c(fp,length(detect))
    T_time<-c(T_time,mean(d_time)) # Record the total execution time when no drift is detected.
    print(T_time)
    mem<-object.size(win1)+object.size(win2)+object.size(result)
    MEM<-c(MEM,as.numeric(mem))
  }
  
  m_fpr<-mean(fp/L)
  sd_fpr<-sd(fp/L)
  
  m_execution_time<-mean(T_time)
  sd_execution_time<-sd(T_time)
 
  m_memory<-mean(MEM)
  sd_memory<-sd(MEM)
  
  df<-data.frame(fpr=paste(m_fpr,'/',sd_fpr),Memory_usage=paste(m_memory,'/',sd_memory),execution_time=paste(m_execution_time,'/',sd_execution_time))
  write.csv(df,file = paste('ww_fp_',win_size,'_',L,'.csv',sep = ''))
}


set.seed(0)
for (win_size in seq(50,200,10)){
  detect(rho = 0.2,sd = 0.2,L = 50000,d = 2, win_size = win_size)
}