# install.packages('plotrix')
# library(plotrix)

jpeg("feature_12_after.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-1
b<-2
drift_time<-416
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:616,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
#plot(data_before, col = cluster_before + 1, pch = 19)
names(data_before)<-c('Water temperature','Turbidity')
names(data_after)<-c('Water temperature','Turbidity')

# names(data_before)<-c('Turbidity','Conductivity')
# names(data_after)<-c('Turbidity','Conductivity')
#plot(data_before,  col = "red",pch = 19,xlim=c(0,0.2),ylim=c(0,0.2),xlab='',ylab='')
plot(data_after,  col = "red",pch = 19,xlim=c(0,0.2),ylim=c(0,0.2),xlab='',ylab='')


par_1<-data_before[cluster_before==1,]
center_1<-c(f1$locality_x_1[1],f1$locality_y_1[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f1$locality_x_2[1],f1$locality_y_2[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f1$locality_x_1[1],
       y = f1$locality_y_1[1],
       pch = 4)
points(x = f1$locality_x_2[1],
       y = f1$locality_y_2[1],
       pch = 4)

dev.off()



# install.packages('plotrix')
# library(plotrix)

jpeg("feature_13_after.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-1
b<-3
drift_time<-401
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:600,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
names(data_before)<-c('Water temperature','Conductivity')
names(data_after)<-c('Water temperature','Conductivity')
plot(data_before, col = "red",pch = 19,xlim=c(0,0.3),ylim=c(0.7,1.0),xlab='',ylab='')
plot(data_after,  col = "red",pch = 19,xlim=c(0,0.3),ylim=c(0.7,1.0),xlab='',ylab='')


par_1<-data_before[cluster_before==1,]
center_1<-c(f2$locality_x_2[1],f2$locality_y_2[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f2$locality_x_1[1],f2$locality_y_1[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f2$locality_x_1[1],
       y = f2$locality_y_1[1],
       pch = 4)
points(x = f2$locality_x_2[1],
       y = f2$locality_y_2[1],
       pch = 4)

dev.off()




jpeg("feature_14_after.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-1
b<-4
drift_time<-401
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:600,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
names(data_before)<-c('Water temperature','pH')
names(data_after)<-c('Water temperature','pH')
#plot(data_before, col = "red",pch = 19,xlim=c(0,0.2),ylim=c(0,0.2),xlab='',ylab='')
plot(data_after,  col = "red",pch = 19,xlim=c(0,0.2),ylim=c(0,0.2),xlab='',ylab='')


par_1<-data_before[cluster_before==1,]
center_1<-c(f3$locality_x_2[1],f3$locality_y_2[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f3$locality_x_1[1],f3$locality_y_1[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f3$locality_x_1[1],
       y = f3$locality_y_1[1],
       pch = 4)
points(x = f3$locality_x_2[1],
       y = f3$locality_y_2[1],
       pch = 4)

dev.off()


jpeg("feature_15_after.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-1
b<-5
drift_time<-418
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:618,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
names(data_before)<-c('Water temperature','Dissolved Oxygen')
names(data_after)<-c('Water temperature','Dissolved Oxygen')
#plot(data_before, col = "red",pch = 19,xlim=c(0,0.15),ylim=c(0.15,0.3),xlab='',ylab='')
plot(data_after,   col = "red",pch = 19,xlim=c(0,0.15),ylim=c(0.15,0.3),xlab='',ylab='')


par_1<-data_before[cluster_before==1,]
center_1<-c(f4$locality_x_1[1],f4$locality_y_1[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f4$locality_x_2[1],f4$locality_y_2[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f4$locality_x_1[1],
       y = f4$locality_y_1[1],
       pch = 4)
points(x = f4$locality_x_2[1],
       y = f4$locality_y_2[1],
       pch = 4)

dev.off()


jpeg("feature_23_before.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-2
b<-3
drift_time<-401
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:600,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
names(data_before)<-c('Turbidity','Conductivity')
names(data_after)<--c('Turbidity','Conductivity')
plot(data_before, col = "red",pch = 19,xlim=c(-0.05,0.1),ylim=c(0.75,0.9),xlab='',ylab='')
#plot(data_after,  col = "red",pch = 19,xlim=c(-0.05,0.1),ylim=c(0.75,0.9),xlab='',ylab='')


par_1<-data_before[cluster_before==1,]
center_1<-c(f5$locality_x_1[1],f5$locality_y_1[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f5$locality_x_2[1],f5$locality_y_2[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f5$locality_x_1[1],
       y = f5$locality_y_1[1],
       pch = 4)
points(x = f5$locality_x_2[1],
       y = f5$locality_y_2[1],
       pch = 4)

dev.off()



jpeg("feature_24_after.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-2
b<-4
drift_time<-401
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:600,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
names(data_before)<-c('Turbidity','pH')
names(data_after)<--c('Turbidity','pH')
#plot(data_before, col = "red",pch = 19,xlim=c(0,0.1),ylim=c(0.1,0.2),xlab='',ylab='')
plot(data_after,  col = "red",pch = 19,xlim=c(0,0.1),ylim=c(0.1,0.2),xlab='',ylab='')


par_1<-data_before[cluster_before==1,]
center_1<-c(f6$locality_x_1[1],f6$locality_y_1[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f6$locality_x_2[1],f6$locality_y_2[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f6$locality_x_1[1],
       y = f6$locality_y_1[1],
       pch = 4)
points(x = f6$locality_x_2[1],
       y = f6$locality_y_2[1],
       pch = 4)

dev.off()





jpeg("feature_25_before.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-2
b<-5
drift_time<-401
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:600,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
names(data_before)<-c('Turbidity','Conductivity')
names(data_after)<-c('Turbidity','Conductivity')
plot(data_before, col = "red",pch = 19,xlim=c(0,0.1),ylim=c(0.16,0.26),xlab='',ylab='')
#plot(data_after,  col = "red",pch = 19,xlim=c(0,0.1),ylim=c(0.16,0.26),xlab='',ylab='')


par_1<-data_before[cluster_before==1,]
center_1<-c(f7$locality_x_1[1],f7$locality_y_1[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f7$locality_x_2[1],f7$locality_y_2[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f7$locality_x_1[1],
       y = f7$locality_y_1[1],
       pch = 4)
points(x = f7$locality_x_2[1],
       y = f7$locality_y_2[1],
       pch = 4)

dev.off()






jpeg("feature_34_after.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-3
b<-4
drift_time<-401
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:600,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
names(data_before)<-c('Conductivity','pH')
names(data_after)<--c('Conductivity','pH')
#plot(data_before, col = "red",pch = 19,xlim=c(0.7,0.9),ylim=c(0,0.2),xlab='',ylab='')
plot(data_after,  col = "red",pch = 19,xlim=c(0.7,0.9),ylim=c(0,0.2),xlab='',ylab='')

par_1<-data_before[cluster_before==1,]
center_1<-c(f8$locality_x_2[1],f8$locality_y_2[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f8$locality_x_1[1],f8$locality_y_1[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f8$locality_x_1[1],
       y = f8$locality_y_1[1],
       pch = 4)
points(x = f8$locality_x_2[1],
       y = f8$locality_y_2[1],
       pch = 4)

dev.off()





jpeg("feature_35_before.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-3
b<-5
drift_time<-401
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:600,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
names(data_before)<-c('Conductivity','Dissolved Oxygen')
names(data_after)<-c('Conductivity','Dissolved Oxygen')
plot(data_before, col = "red",pch = 19,xlim=c(0.7,0.9),ylim=c(0.1,0.3),xlab='',ylab='')
#plot(data_after,  col = "red",pch = 19,xlim=c(0.7,0.9),ylim=c(0.1,0.3),xlab='',ylab='')

par_1<-data_before[cluster_before==1,]
center_1<-c(f9$locality_x_1[1],f9$locality_y_1[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f9$locality_x_2[1],f9$locality_y_2[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f9$locality_x_1[1],
       y = f9$locality_y_1[1],
       pch = 4)
points(x = f9$locality_x_2[1],
       y = f9$locality_y_2[1],
       pch = 4)

dev.off()


f

jpeg("feature_45_before.jpeg", width = 4, height = 4.5, units = 'in', res = 300)
a<-4
b<-5
drift_time<-401
stream<-read.csv('water_data_scaled.csv')[,-1][,c(a,b)]
data_before<-stream[0:200,]
data_after<-stream[drift_time:600,]
cluster_before<-readRDS(paste(drift_time,'_feature_',a,'_',b,"_assignment.RData",sep=''))
data_after<-read.csv(paste('win2_',drift_time,'_feature_',a,'_',b,'.csv',sep = ''))
names(data_before)<-c('pH','Dissolved Oxygen')
names(data_after)<-c('pH','Dissolved Oxygen')
plot(data_before, col = "red",pch = 19,xlim=c(0.1,0.2),ylim=c(0.16,0.26),xlab='',ylab='')
#plot(data_after,  col = "red",pch = 19,xlim=c(0.1,0.2),ylim=c(0.16,0.26),xlab='',ylab='')

par_1<-data_before[cluster_before==1,]
center_1<-c(f10$locality_x_1[1],f10$locality_y_1[1])
radius_1<-mean(apply(par_1,1,euc.dist<-function(x1, x2=center_1) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_1[1],center_1[2],radius_1)


par_2<-data_before[cluster_before==2,]
center_2<-c(f10$locality_x_2[1],f10$locality_y_2[1])
radius_2<-mean(apply(par_2,1,euc.dist<-function(x1, x2=center_2) sqrt(sum((x1 - x2) ^ 2))))
draw.circle(center_2[1],center_2[2],radius_2)

points(x = f10$locality_x_1[1],
       y = f10$locality_y_1[1],
       pch = 4)
points(x = f10$locality_x_2[1],
       y = f10$locality_y_2[1],
       pch = 4)

dev.off()










