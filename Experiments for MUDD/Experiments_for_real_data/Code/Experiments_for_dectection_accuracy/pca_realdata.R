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
  detect<-NULL
  delta<-0.005
  alpha<-1-0.0001
  kl_count<-0
  kl_mean<-0
  dif<-0
  
  pca <- prcomp(win1, center = TRUE,scale. = TRUE)
  # Determine the number (k) of PCS to use.
  eigs <- pca$sdev^2
  prop_of_var<-eigs / sum(eigs)
  cum_prop<-cumsum(prop_of_var)
  k<-which(cum_prop>=0.999)[1]
  
  # Project data in win1 and win2 to PC space
  new_win1<-as.data.frame(scale(win1, pca$center, pca$scale) %*% pca$rotation)
  new_win1<-new_win1[,1:k]
  
  for (data_point in 201:nrow(stream)){
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
    print(end-start)
  }
  
  
  range_1<-seq(3000,93000,3000)
  range_2<-range_1+200
  tp<-0
  DD<-NULL
  fp<-0
  flag<-rep(FALSE,30)
  for (i in detect){
    f<-FALSE
    print(i)
    for(j in 1:30){
      if (i %in% range_1[j]:range_2[j]){
        f<-TRUE
        if (!flag[j]){
          tp<-tp+1
          dd<-i-range_1[j]
          print(i)
          print(range_1[j])
          DD<-c(DD,dd)
          flag[j]<-TRUE
          break
        }else{
          fp<-fp+1
        }
      }
      
    }
    if (!f){
      fp<-fp+1
    }
  }
  
  if(is.null(DD)){
    m_dd<-0
    s_dd<-0
  }else{
    m_dd<-mean(DD)
    s_dd<-sd(DD)
  }
  
  fn<-30-tp
  precision<-tp/(tp+fp)
  recall<-tp/(tp+fn)
  f1<-2*(precision*recall)/(precision+recall)
  
  df<-data.frame(tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=f1,m_dd=m_dd,s_dd=s_dd)
  
  write.csv(df,file = paste('pca_',stream_name,'.csv',sep = ''))
  
}
