library(gamm4)
library(MuMIn)

data=read.csv("/home/max/Documents/DRD/ES_mlm_data.csv")

#####build mixed model function for abcd####
mixed_model=function(x,y,covs,data){
  if (x != y){
  stat_holder <- data.frame()
  stat_names <- c('B','SE','t','p','R2')
  for (k in stat_names) stat_holder[k] <- as.double()
  
  form_cov_only <- formula(paste(y, "~", paste(covs, collapse="+")))
  form <- formula(paste(y, "~", x, "+", paste(covs, collapse="+")))
  model <- gamm4(form, data=data, random =~(1|mri_info_device.serial.number/rel_family_id) )
  model2 <- gamm4(form_cov_only, data=data, random =~(1|mri_info_device.serial.number/rel_family_id))
  r2_delta = round(as.numeric(r.squaredLR(model$mer,model2$mer)),5)
  sg<-summary(model$gam)
  
  for (statnum in 1:4){
    stat_holder[1,statnum]<-sg$p.table[2,statnum]
  }
  stat_holder[1,5]<-r2_delta
  return(stat_holder)
  }
}

ivs <- readLines('/home/max/Documents/DRD/ES_cov_varlist.txt')
xs <- ivs[1:3]
ys <- ivs[c(2,24)]
covs <- c('age','sex','high.educ.bl','household.income.bl','married.bl','hisp',"race.6level")

stat_list <- list()

for (y in ys){
stat_list2 <- data.frame()
stat_list2<-(lapply(x, mixed_model, covs=covs,y=y, data=data))
stat_list <- rbind(stat_list,stat_list2)
}

stat_matrix <- data.frame()
stat_names <- c('var1','var2','B','SE','t','p','R2')
for (k in stat_names) stat_matrix[k] <- as.double()
for (i in 1:length(stat_list)){stat_matrix[i,]=stat_list[[i]]}
for (n in 1:length(ys)) row.names(stat_matrix)[n] <- ys[n]



stat_holder <- data.frame()
stat_names <- c('var1','var2','B','SE','t','p','R2')
for (k in stat_names) stat_holder[k] <- as.double()
i = 1
for (x in xs) {
  for (y in ys) {
    if (x != y){
    form_cov_only <- formula(paste(y, "~", paste(covs, collapse="+")))
    form <- formula(paste(y, "~", x, "+", paste(covs, collapse="+")))
    model <- gamm4(form, data=data, random =~(1|mri_info_device.serial.number/rel_family_id) )
    model2 <- gamm4(form_cov_only, data=data, random =~(1|mri_info_device.serial.number/rel_family_id))
    r2_delta = round(as.numeric(r.squaredLR(model$mer,model2$mer)),5)
    sg<-summary(model$gam)
    stat_holder[i,1] <- x
    stat_holder[i,2] <- y
    for (statnum in 1:4){
      stat_holder[i,(statnum+2)]<-sg$p.table[2,statnum]
      }
    stat_holder[i,7]<-r2_delta
    i =+ 1
    }
  }
}

write.csv(stat_holder,"/home/max/Documents/DRD/ES_MLM_stats.csv")
