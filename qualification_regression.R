library(DescTools)

data = read.csv("/home/max/Documents/DRD/owens_updated.csv")
par(mar = c(1, 1, 1, 1))
dim(data);names(data)
#data=data[!(data$within_instrument==1 & data$within_reporter==0),]
dim(data);names(data)

R = data$cor
zscore = FisherZ(R)
data$z = zscore*sqrt(data$n-3)
data$z_log = log(data$z)
hist(data$z_log)
summary(lm(z_log~within_domain+within_instrument+within_reporter, data=data))
summary(lm(z_log~within_domain, data=data))
summary(lm(z_log~within_instrument, data=data))
summary(lm(z_log~within_reporter, data=data))

# technically for p-values in the regression I think you would need to do a bootstrap...
