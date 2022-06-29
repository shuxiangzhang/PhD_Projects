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

td.test<-function(x,y){
  ranked_x<-x[order(x,decreasing = T)]
  ranked_y<-y[order(x,decreasing = T)]
  if (ranked_x[1]>ranked_y[1]){
    c_1<-length(ranked_x[ranked_x>ranked_y[1]])
    c_2<-length(ranked_y[ranked_y<ranked_x[length(ranked_x)]])
    c<-c_1+c_2
  }else{
    c_1<-length(ranked_y[ranked_y>ranked_x[1]])
    c_2<-length(ranked_x[ranked_x<ranked_y[length(ranked_y)]])
    c<-c_1+c_2
  }
  return (c>=7)
}

set.seed(0)
L = 50000
  rho <- 0.2
  sd <- 0.2
  fp<-NULL
  MEM<-NULL
  T_time<-NULL
  for (i in seq(30)){
    d_time<-NULL
    mem<-NULL
    stream<-stream_generator(rho,sd,L)
    win1<-stream[1:100,]
    win2<-stream[101:200,]
    detect<-NULL
    flag<-TRUE
    counter<-0
    latest_point<-200
    for (j in 201:L){
      print(j)
      start_time = Sys.time()
      win2<-win2[-1,]
      win2<-rbind(win2,stream[j,])
      if (flag==TRUE){
        if (j-latest_point == 100){
          latest_point<-j
          ave_1<-colMeans(win1)
          ave_2<-colMeans(win2)
          f_1<-apply(win1,1,function(x) dist(rbind(x,ave_1)))
          f_2<-apply(win2,1,function(x) dist(rbind(x,ave_1)))
          result<-td.test(f_1,f_2)
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
        if (counter==100){
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
  write.csv(df,file = paste('uni_td_fp_',L,'.csv',sep = ''))
