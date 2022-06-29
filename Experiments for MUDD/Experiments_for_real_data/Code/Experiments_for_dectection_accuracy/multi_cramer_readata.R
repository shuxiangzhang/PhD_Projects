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

stream_names<-c('el_g.csv','el_s.csv','asc_g.csv','asc_s.csv','cyc_g.csv','cyc_s.csv','des_g.csv','des_s.csv','iro_g.csv','iro_s.csv','lpine_g.csv','lpine_s.csv','spruce_g.csv','spruce_s.csv','vac_g.csv','vac_s.csv')
for (stream_name in stream_names){
stream<-read.table(stream_name,stringsAsFactors = FALSE)
stream<-as.matrix(stream)
win1<-stream[1:100,]
win2<-stream[101:200,]
detect<-NULL
count<-0
flag<-TRUE 
for (j in 201:nrow(stream)){
  start<-Sys.time()
  print(j)
  # Break the loop if no drift was detected after the whole sliding window has entered the new distribution.
  win2<-win2[-1,]
  win2<-rbind(win2,stream[j,])
  win<-rbind(win1,win2)
  if (flag){
    result<-cramer.test(win1,win2,replicates = 1000)
    p_value<-result$p.value
    if (p_value<0.05){
      detect<-c(detect,j)
      print(paste('Drift was detected at:',j))
      count<-0
      flag<-FALSE
    }
    
  }else{
    count<-count+1
    if(count==200){
      win1<-stream[(j-199):(j-100),]
      flag<-TRUE
    }
    end<-Sys.time()
    print(end-start)
  }}

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

write.csv(df,file = paste('multi_cramer_',stream_name,'.csv',sep = ''))
}