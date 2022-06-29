library(pryr)
library(GSAR)
library(dplyr)
library(cramer)
library(MASS)


stream_generator<-function(rho,mu,L,step_size,d){
  sd<-0.2
  mu<-rep(mu,d)
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
  print(Sigma)
  s1<-mvrnorm((L-1000), mu, Sigma )
  
  sd<-sd+step_size
  for(i in seq(d)){
    for(j in seq(d)){
      if (i==j){
        Sigma[i,j]<-sd^2
      }else{
        Sigma[i,j]<-sd^2*rho
      }
    }
  }
  print(Sigma)
  s2<-mvrnorm(1000, mu, Sigma )
  stream<-rbind(s1,s2)
  return(stream)
}

detect<-function(rho,mu,L,step_size,d){
  
  TP<-0
  DD<-NULL
  D_location<-list()
  delta<-0.005
  alpha<-1-0.0001
  
  for (w in seq(30)){
    dd<-NULL
    kl_count<-0
    kl_mean<-0
    dif<-0
    stream<-stream_generator(rho,mu,L,step_size,d)
    win1<-stream[(L-1000-2*100+1):(L-1000-100),]
    win2<-stream[(L-1000-100+1):(L-1000),]
    
    pca <- prcomp(win1, center = TRUE,scale. = TRUE)
    # Determine the number (k) of PCS to use.
    eigs <- pca$sdev^2
    prop_of_var<-eigs / sum(eigs)
    cum_prop<-cumsum(prop_of_var)
    k<-which(cum_prop>=0.999)[1]
    
    # Project data in win1 and win2 to PC space
    new_win1<-as.data.frame(scale(win1, pca$center, pca$scale) %*% pca$rotation)
    new_win1<-new_win1[,1:k]
    
    
    for (n in (L-999):L){
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
  write.csv(df,file = paste('PCA-CD_sd_abrupt','_',d,'_',L,'_','.csv',sep = ''))
}



set.seed(0)
detect(rho = 0.2, mu = 0.2, L = 50000, step_size = 0.5,d = 2)


