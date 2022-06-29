install.packages('stream')
install.packages('ggpubr')
install.packages('factoextra')
install.packages('philentropy')
install.packages('MASS')
install.packages('dplyr')
install.packages('cramer')
install.packages('LICORS')
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("GSAR")
install.packages('pryr')
library(LICORS)
library(philentropy)
library(stream)
library(ggpubr)
library(factoextra)
library(pryr)
library(GSAR)
library(dplyr)
library(cramer)
library(MASS)

stream_generator<-function(dim,m,k){
  
  ex<-'MGC_Static(dens=1, par=0.5, center=rep(1,dim))'
  for (x in seq(k-1)){
    ex<-paste(ex,',MGC_Static(dens=1, par=0.5, center=rep(',(1+x*7),',dim))')
  }
  ex<-paste('DSD_MG(dim = dim,',ex,')')
  stream_1<- eval(parse(text = ex))
  stream_1<-get_points(stream_1,k*500)
  
  ex<-'MGC_Static(dens=1, par=0.5, center=rep(1,dim))'
  if(k>2){
    for (x in seq(k-2)){
    ex<-paste(ex,',MGC_Static(dens=1, par=0.5, center=rep(',(1+x*7),',dim))')
  }}
  last<-paste(',MGC_Static(dens=1, par=0.5, center=rep(',(m+1+(k-1)*7),',dim))')
  ex<-paste('DSD_MG(dim = dim,',ex,last,')')
  stream_2<- eval(parse(text = ex))
  stream_2<-get_points(stream_2,k*500)
  stream<-rbind(stream_1,stream_2)
}





table_for_drift_types<-data.frame(drift_code=c(1,2,3,4,5),drift_name=c('shrinkness or skewness','partially moving','completely moving','exapnding','falsely staying'))

detect<-function(dim,m,k){
  final_result<-data.frame(matrix(nrow= 1,ncol = 4))
  for (b in seq(30)){
    batch_num<-b
    stream<-stream_generator(dim,m,k)
    win1<-stream[1:(k*100),]
    win2<-stream[((k*100)+1):2*(k*100),]
    flag<-TRUE
    counter<-0
    for (j in (2*(k*100)+1):nrow(stream)){
      print(j)
      win2<-win2[-1,]
      win2<-rbind(win2,stream[j,])
      if (flag){
        ave_1<-colMeans(win1)
        ave_2<-colMeans(win2)
        f_1<-apply(win1,1,function(x) dist(rbind(x,ave_1)))
        f_2<-apply(win2,1,function(x) dist(rbind(x,ave_1)))
        result<-ks.test(f_1,f_2)
        p_value<-result$p.value
        if (p_value<0.05){
          drift_time<-j
          print(paste('Drift was detected at:',j))
          flag<-FALSE
        }
      }else{
        counter<-counter+1
        if (counter==(k*100)){
          res.km<-kmeanspp(win1,k)
          assignment<-res.km$cluster
          names(assignment)<-NULL
          original_par_size<-res.km$size
          original_data_per_partition<-list()
          radius_per_partition<-list()
          remaining_par_size<-list()
          remaining_data_per_partition<-list()
          
          for (i in seq(k)){
            x<-win1[which(assignment==i),]
            original_data_per_partition[[i]]<-x
            y<-res.km$centers[i,]
            radius<-max(apply(x,1,euc.dist<-function(x1, x2=y) sqrt(sum((x1 - x2) ^ 2))))
            radius_per_partition[[i]]<-radius
            remaining_par_size[[i]]<-0
            remaining_data_per_partition[[i]]<-data.frame(matrix(nrow=1,ncol = dim))
          }
          
          for (j in 1:nrow(win2)){
            min_dist<-min(apply(res.km$centers,1,euc.dist<-function(x1, x2=win2[j,]) sqrt(sum((x1 - x2) ^ 2))))
            partition_index<-which.min(apply(res.km$centers,1,euc.dist<-function(x1, x2=win2[j,]) sqrt(sum((x1 - x2) ^ 2))))
            if(min_dist<=radius_per_partition[[partition_index]]){
              remaining_par_size[[partition_index]]<-remaining_par_size[[partition_index]]+1
              remaining_data_per_partition[[partition_index]]<-rbind(remaining_data_per_partition[[partition_index]],win2[j,])
            }
          }
          remaining_par_size<-unlist(remaining_par_size)
          
          intensity<-abs((original_par_size-remaining_par_size))/original_par_size
          
          localityRatio<-NULL
          for (i in seq(k)){
            centroid_remaining<-colMeans(remaining_data_per_partition[[i]],na.rm = TRUE)
            if (is.nan(centroid_remaining[1])){
              localityRatio<-c(localityRatio,1)
            }else{
              localityRatio<-c(localityRatio,euc.dist(centroid_remaining,res.km$centers[i,])/radius_per_partition[[i]])
            }
          }
          print(intensity)
          print(localityRatio)
          print(res.km$centers)
          print(original_par_size)
          print(remaining_par_size)
          # Make the partition number consistent
          first_coordinate_of_center<-unname(res.km$centers[,1])
          corresponding_partion_for_centroid<-order(first_coordinate_of_center)
          locality<-res.km$centers[corresponding_partion_for_centroid,]
          localityRatio<-localityRatio[corresponding_partion_for_centroid]
          intensity<-intensity[corresponding_partion_for_centroid]
          
          # Determine drift type based on the intensity and localityRatio
          for (par_num in seq(k)){      
            partition_num<-par_num
            real_num<-corresponding_partion_for_centroid[par_num]
            if (intensity[par_num]<=0.2){
              ave_1<-colMeans(original_data_per_partition[[real_num]])
              ave_2<-colMeans(remaining_data_per_partition[[real_num]])
              f_1<-apply(original_data_per_partition[[real_num]],1,function(x) dist(rbind(x,ave_1)))
              f_2<-apply(remaining_data_per_partition[[real_num]],1,function(x) dist(rbind(x,ave_1)))
              result<-ks.test(f_1,f_2)
              p_value<-result$p.value
              print(p_value)
              if (p_value<0.05){
                drift_type<-1
                final_result<-rbind(final_result,c(partition_num,drift_type,drift_time,batch_num))
                print(paste('shrinks or skewness drift for partition:', locality[par_num,1],locality[par_num,2]))
              }else{
                drift_type<-0
                final_result<-rbind(final_result,c(partition_num,drift_type,drift_time,batch_num))
                print(paste('no drift for partition:', locality[par_num,1],locality[par_num,2]))
              }
              
            }else if (intensity[par_num]>0.2 && intensity[par_num]<0.9 && localityRatio[par_num]>0.24&& localityRatio[par_num]<0.9){
              drift_type<-2
              final_result<-rbind(final_result,c(partition_num,drift_type,drift_time,batch_num))
              print(paste('partially moving for partition:', locality[par_num,1],locality[par_num,2]))
            }else if (intensity[par_num]>=0.9 || localityRatio[par_num]>=0.9){
              drift_type<-3
              final_result<-rbind(final_result,c(partition_num,drift_type,drift_time,batch_num))
              print(paste('completely moving for partition:', locality[par_num,1],locality[par_num,2]))
            }else if (intensity[par_num]>0.2 && intensity[par_num]<0.9 && localityRatio[par_num]<=0.24){
              ave_1<-colMeans(original_data_per_partition[[real_num]])
              ave_2<-colMeans(remaining_data_per_partition[[real_num]])
              f_1<-apply(original_data_per_partition[[real_num]],1,function(x) dist(rbind(x,ave_1)))
              f_2<-apply(remaining_data_per_partition[[real_num]],1,function(x) dist(rbind(x,ave_1)))
              result<-ks.test(f_1,f_2)
              p_value<-result$p.value
              if (p_value<0.05){
                drift_type<-4
                final_result<-rbind(final_result,c(partition_num,drift_type,drift_time,batch_num))
                print(paste('Expanding for partition:', locality[par_num,1],locality[par_num,2]))
              }else{
                drift_type<-5
                final_result<-rbind(final_result,c(partition_num,drift_type,drift_time,batch_num))
                print(paste('Falsely staying for partition:', locality[par_num,1],locality[par_num,2]))
              }
            }else{
              drift_type<-6
              final_result<-rbind(final_result,c(partition_num,drift_type,drift_time,batch_num))
              print(paste('No such type', locality[par_num,1],locality[par_num,2]))
            }
          }
          win1<-win2
        }
        if (counter==2*(k*100)){
          flag<-TRUE
          counter<-0
        }
      }
      
    }
    print(paste('This is run:',batch_num))
  }
  colnames(final_result)<-c('partition_num','drift_type','drift_time','batch_num')
  final_result<-final_result[-1,]
  final_result<-final_result[order(final_result$partition_num),]
  count_partition<-as.numeric(table(final_result$partition_num))
  ground_truth<-rep(c(rep(0,(k-1)),2),count_partition)
  final_result$ground_truth<-ground_truth
  write.csv(final_result,paste(k,'_cluster_final_result_type_2.csv'),row.names = FALSE)
}

set.seed(0)
for(k in seq(2,10,1)){
  detect(2,1,k)
}




