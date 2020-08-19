# DEAP model 2017
###SET THESE BEFORE RUNNING#####
vars=readLines("/home/max/Documents/DRD/es_varlist_622.txt")
strat_vars=c("rel_family_id","mri_info_device.serial.number")
allvars=c('src_subject_id','eventname',strat_vars,vars)
data =  readRDS( paste0("/home/max/Documents/linear_mixed_model_abcd/nda2.0.1.Rds"))
backup_data=data
data=backup_data
data = data[allvars]
data <-data[ which(data$eventname=='baseline_year_1_arm_1'),]
data2=data

class_list <- vector(mode = "list", length = length(data))
i=1
for (v in data){
  class_list[[i]]=data.class(v)
  i=i+1
}
class_list2=lapply(data,data.class)

facs=lapply(Filter(is.factor,data), levels)
cols <- names(facs[4:length(facs)])
fac_cols=cols[c(-13:-35,-38)]
data[fac_cols] <- lapply(data[fac_cols], as.numeric)
data3<-data[fac_cols]

class_list3=lapply(data,data.class)
facs3=lapply(Filter(is.factor,data3), levels)

####remove 1 sibling#######

write.csv(data,"/home/max/Documents/DRD/es_data.csv",row.names = FALSE)

data$ksads_back_c_det_susp_p[1:20]
data2$ksads_back_c_det_susp_p[1:20]
unique(data$ksads_back_c_det_susp_p)
unique(data2$ksads_back_c_det_susp_p)
