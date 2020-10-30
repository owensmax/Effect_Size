# DEAP model 2017
###SET THESE BEFORE RUNNING#####
vars=readLines("/home/max/Documents/DRD/es_varlist_622.txt")
strat_vars=c("rel_family_id","mri_info_device.serial.number")
allvars=c('src_subject_id','eventname',strat_vars,vars)
data =  readRDS( paste0("/home/max/Documents/linear_mixed_model_abcd/nda2.0.1.Rds"))
backup_data=data
data=backup_data
data <-data[ which(data$eventname=='baseline_year_1_arm_1'),]

library('psych')
library(dplyr)
library(hash)

cas=data.frame()

a=alpha(data.frame(lapply(select(data,upps_7, upps_11, upps_17,upps_20), as.numeric)))
cas[1,1]=a$total$raw_alpha
colnames(cas)[1]='upps_ss_negative_urgency'

a=alpha(data.frame(lapply(select(data,upps_6, upps_16, upps_23,upps_28), as.numeric)))
cas[1,2]=a$total$raw_alpha
colnames(cas)[2]='upps_ss_lack_of_planning'

a=alpha(data.frame(lapply(select(data,upps_12, upps_18, upps_21,upps_27), as.numeric)))
cas[1,3]=a$total$raw_alpha
colnames(cas)[3]='upps_ss_sensation_seeking'

a=alpha(data.frame(lapply(select(data,upps_35, upps_36, upps_37,upps_39), as.numeric)))
cas[1,4]=a$total$raw_alpha
colnames(cas)[4]='upps_ss_positive_urgency'

a=alpha(data.frame(lapply(select(data,upps_15, upps_19, upps_22,upps_24), as.numeric)))
cas[1,5]=a$total$raw_alpha
colnames(cas)[5]='upps_ss_lack_of_perseverance'

pds=data[grepl("prodrom_psych_", names(data))]
a=alpha(data.frame(lapply(select(data,prodrom_psych_1,	prodrom_psych_2,	prodrom_psych_3,	prodrom_psych_4,	prodrom_psych_5,	prodrom_psych_6,	prodrom_psych_7,	prodrom_psych_8,	prodrom_psych_9,	prodrom_psych_10,	prodrom_psych_11,	prodrom_psych_12,	prodrom_psych_13,	prodrom_psych_14,	prodrom_psych_15,	prodrom_psych_16,	prodrom_psych_17,	prodrom_psych_18,	prodrom_psych_19,	prodrom_psych_20,	prodrom_psych_21), as.numeric)))
cas[1,6]=a$total$raw_alpha
colnames(cas)[6]='prodrom_psych_ss_severity_score'

bisbas_rr=alpha(data.frame(lapply(select(data,bisbas_8,bisbas_9,bisbas_10,bisbas_11,bisbas_12), as.numeric)))
bisbas_drive=alpha(data.frame(lapply(select(data,bisbas_13,bisbas_14,bisbas_15,bisbas_16), as.numeric)))  
bisbas_funseeking=alpha(data.frame(lapply(select(data,bisbas_17,bisbas_18,bisbas_19,bisbas_20), as.numeric)))
cas[1,7]=bisbas_rr$total$raw_alpha
colnames(cas)[7]='bisbas_ss_bas_rr'
cas[1,8]=bisbas_drive$total$raw_alpha
colnames(cas)[8]='bisbas_ss_bas_drive'
cas[1,9]=bisbas_funseeking$total$raw_alpha
colnames(cas)[9]='bisbas_ss_bas_fs'

ehi=alpha(data.frame(lapply(select(data,ehi_1b,ehi_2b,ehi_3b,ehi_4b), as.numeric)))
cas[1,10]=ehi$total$raw_alpha
colnames(cas)[10]='ehi_ss_score'

macv_fs=alpha(data.frame(lapply(select(data,macvs_2_p,macvs_7_p,macvs_12_p,macvs_16_p,macvs_21_p,macvs_26_p), as.numeric)))
macv_fr=alpha(data.frame(lapply(select(data,macvs_24_p,macvs_9_p,macvs_18_p,macvs_23_p,macvs_27_p), as.numeric)))
macv_fo=alpha(data.frame(lapply(select(data,macvs_3_p,macvs_8_p,macvs_13_p,macvs_17_p,macvs_22_p,macvs_26_p), as.numeric)))
macv_isr=alpha(data.frame(lapply(select(data,macvs_5_p,macvs_10_p,macvs_14_p,macvs_19_p,macvs_24_p), as.numeric)))
macv_p_ss_r=alpha(data.frame(lapply(select(data,macvs_1_p,macvs_6_p,macvs_11_p,macvs_15_p,macvs_20_p,macvs_25_p,macvs_28_p), as.numeric)))
cas[1,11]=macv_fs$total$raw_alpha
cas[1,12]=macv_fo$total$raw_alpha
cas[1,13]=macv_isr$total$raw_alpha
cas[1,14]=macv_p_ss_r$total$raw_alpha
cas[1,15]=macv_fr$total$raw_alpha
colnames(cas)[11]='macvs_ss_fs_p'
colnames(cas)[12]='macvs_ss_fo_p'
colnames(cas)[13]='macvs_ss_isr_p'
colnames(cas)[14]='macvs_ss_r_p'
colnames(cas)[15]='macvs_ss_fr_p'

fes_ss_fc=alpha(data.frame(lapply(select(data,fes_q1,fes_q2,fes_q3,fes_q4,fes_q5,fes_q6,fes_q7,fes_q8,fes_q9), as.numeric)))
fes_ss_fc_p=alpha(data.frame(lapply(select(data,fes_1_p,fes_2r_p,fes_3_p,fes_4r_p,fes_5_p,fes_6_p,fes_7r_p,fes_8_p,fes_9_p), as.numeric)))
cas[1,16]=fes_ss_fc$total$raw_alpha
cas[1,17]=fes_ss_fc_p$total$raw_alpha
colnames(cas)[16]='fes_ss_fc'
colnames(cas)[17]='fes_ss_fc_p'

prosocial_p=alpha(data.frame(lapply(select(data,prosocial_q1_p,prosocial_q2_p,prosocial_q3_p), as.numeric)))
prosocial_y=alpha(data.frame(lapply(select(data,prosocial_q1,prosocial_q2,prosocial_q3), as.numeric)))
cas[1,18]=prosocial_p$total$raw_alpha
cas[1,19]=prosocial_y$total$raw_alpha
colnames(cas)[18]='prosocial_ss_mean_p'
colnames(cas)[19]='prosocial_ss_mean'

caregiver_acceptance=alpha(data.frame(lapply(select(data,crpbi_acceptance_studycaregiver1,crpbi_acceptance_studycaregiver2,crpbi_acceptance_studycaregiver3,crpbi_acceptance_studycaregiver4,crpbi_acceptance_studycaregiver5), as.numeric)))
cas[1,20]=caregiver_acceptance$total$raw_alpha
colnames(cas)[20]='crpbi_acceptance_ss_studycaregiver'

srpf_ses=alpha(data.frame(lapply(select(data,school_risk_phenx_2,school_risk_phenx_3,school_risk_phenx_4,school_risk_phenx_5,school_risk_phenx_6,school_risk_phenx_7), as.numeric)))
srpf_iiss=alpha(data.frame(lapply(select(data,school_risk_phenx_8,school_risk_phenx_9,school_risk_phenx_10,school_risk_phenx_12), as.numeric)))
srpf_dfs=alpha(data.frame(lapply(select(data,school_risk_phenx_15,school_risk_phenx_17), as.numeric))) #only 2 items maybe drop?
cas[1,21]=srpf_ses$total$raw_alpha
colnames(cas)[21]='school_risk_phenx_ss_ses'
cas[1,22]=srpf_iiss$total$raw_alpha
colnames(cas)[22]='school_risk_phenx_ss_iiss'
cas[1,23]=srpf_dfs$total$raw_alpha
colnames(cas)[23]='school_risk_phenx_ss_dfs'

neighb_phenx_ss_mean_p=alpha(data.frame(lapply(select(data,neighb_phenx_1r_p,neighb_phenx_2r_p,neighb_phenx_3r_p), as.numeric)))
cas[1,24]=neighb_phenx_ss_mean_p$total$raw_alpha
colnames(cas)[24]='neighb_phenx_ss_mean_p'

via_p_ss_hc=alpha(data.frame(lapply(select(data,via_accult_q2_p,via_accult_q4_p,via_accult_q6_p,via_accult_q8_p,via_accult_q10_p,via_accult_q12_p,via_accult_q14_p,via_accult_q16_p), as.numeric)))
via_p_ss_amer=alpha(data.frame(lapply(select(data,via_accult_q3_p,via_accult_q5_p,via_accult_q7_p,via_accult_q9_p,via_accult_q11_p,via_accult_q13_p,via_accult_q15_p,via_accult_q17_p), as.numeric)))
cas[1,25]=via_p_ss_hc$total$raw_alpha
colnames(cas)[25]='via_accult_ss_hc_p'
cas[1,26]=via_p_ss_amer$total$raw_alpha
colnames(cas)[26]='via_accult_ss_amer_p'

meim_ss_total_p=alpha(data.frame(lapply(select(data,meim_1_p,meim_4_p,meim_5_p), as.numeric)))
cas[1,27]=meim_ss_total_p$total$raw_alpha
colnames(cas)[27]='meim_ss_total_p'

sleep_dist_shy=alpha(data.frame(lapply(select(data,sleep_9_p,sleep_16_p), as.numeric)))
sleep_dist_does=alpha(data.frame(lapply(select(data,sleep_22_p,sleep_23_p,sleep_24_p,sleep_25_p,sleep_26_p), as.numeric)))
sleep_dist_swtd=alpha(data.frame(lapply(select(data,sleep_6_p,sleep_7_p,sleep_8_p,sleep_12_p,sleep_18_p,sleep_19_p), as.numeric)))
sleep_dist_da=alpha(data.frame(lapply(select(data,sleep_17_p,sleep_20_p,sleep_21_p), as.numeric)))
sleep_dist_sbd=alpha(data.frame(lapply(select(data,sleep_13_p,sleep_14_p,sleep_15_p), as.numeric)))
sleep_dist_dims=alpha(data.frame(lapply(select(data,sleep_1_p,sleep_2_p,sleep_3_p,sleep_4_p,sleep_5_p,sleep_10_p,sleep_11_p), as.numeric)))
cas[1,28]=sleep_dist_shy$total$raw_alpha
cas[1,29]=sleep_dist_does$total$raw_alpha
cas[1,30]=sleep_dist_swtd$total$raw_alpha
cas[1,31]=sleep_dist_da$total$raw_alpha
cas[1,32]=sleep_dist_sbd$total$raw_alpha
cas[1,33]=sleep_dist_dims$total$raw_alpha
colnames(cas)[28]='sleep_dist_shy'
colnames(cas)[29]='sleep_dist_does'
colnames(cas)[30]='sleep_dist_swtd'
colnames(cas)[31]='sleep_dist_da'
colnames(cas)[32]='sleep_dist_sbd'
colnames(cas)[33]='sleep_dist_dims'
sleep_all=alpha(data.frame(lapply(select(data,sleep_1_p,sleep_2_p,sleep_3_p,sleep_4_p,sleep_5_p,sleep_10_p,sleep_11_p,sleep_13_p,sleep_14_p,sleep_15_p,sleep_6_p,sleep_7_p,sleep_8_p,sleep_12_p,sleep_18_p,sleep_19_p,sleep_17_p,sleep_20_p,sleep_21_p,sleep_22_p,sleep_23_p,sleep_24_p,sleep_25_p,sleep_26_p,sleep_9_p,sleep_16_p), as.numeric)))
cas[1,28]=sleep_all$total$raw_alpha
colnames(cas)[28]='sleep_ss_total_p'

su_caff_ss_sum_calc=alpha(data.frame(lapply(select(data,su_caff_intake_1,su_caff_intake_3,su_caff_intake_4,su_caff_intake_6,su_caff_intake_9), as.numeric)))
cas[1,34]=su_caff_ss_sum_calc$total$raw_alpha
colnames(cas)[34]='su_caff_ss_sum_calc'

cbcl_scr_07_ocd_r=alpha(data.frame(lapply(select(data,cbcl_q09_p,cbcl_q31_p,cbcl_q32_p,cbcl_q52_p,cbcl_q66_p,cbcl_q84_p,cbcl_q85_p,cbcl_q112_p), as.numeric)))	
cbcl_scr_07_sct_r=alpha(data.frame(lapply(select(data,cbcl_q13_p,cbcl_q17_p,cbcl_q80_p,cbcl_q102_p), as.numeric)))	
cbcl_scr_07_stress_r=alpha(data.frame(lapply(select(data,cbcl_q03_p,cbcl_q08_p,cbcl_q09_p,cbcl_q11_p,cbcl_q31_p,cbcl_q34_p,cbcl_q45_p,cbcl_q47_p,cbcl_q50_p,cbcl_q52_p,cbcl_q69_p,cbcl_q87_p,cbcl_q103_p,cbcl_q111_p), as.numeric)))		
cbcl_scr_syn_aggressive_r=alpha(data.frame(lapply(select(data,cbcl_q03_p,cbcl_q16_p,cbcl_q19_p,cbcl_q20_p,cbcl_q21_p,cbcl_q22_p,cbcl_q23_p,cbcl_q37_p,cbcl_q57_p,cbcl_q68_p,cbcl_q86_p,cbcl_q87_p,cbcl_q88_p,cbcl_q89_p,cbcl_q94_p,cbcl_q95_p,cbcl_q97_p,cbcl_q104_p), as.numeric)))
cbcl_scr_syn_anxdep_r=alpha(data.frame(lapply(select(data,cbcl_q14_p,cbcl_q29_p,cbcl_q30_p,cbcl_q31_p,cbcl_q32_p,cbcl_q33_p,cbcl_q35_p,cbcl_q45_p,cbcl_q50_p,cbcl_q52_p,cbcl_q71_p,cbcl_q91_p,cbcl_q112_p), as.numeric)))			
cbcl_scr_syn_attention_r=alpha(data.frame(lapply(select(data,cbcl_q01_p,cbcl_q04_p,cbcl_q08_p,cbcl_q10_p,cbcl_q13_p,cbcl_q17_p,cbcl_q41_p,cbcl_q61_p,cbcl_q78_p,cbcl_q80_p), as.numeric)))
cbcl_scr_syn_rulebreak_r=alpha(data.frame(lapply(select(data,cbcl_q02_p,cbcl_q26_p,cbcl_q28_p,cbcl_q39_p,cbcl_q43_p,cbcl_q63_p,cbcl_q67_p,cbcl_q72_p,cbcl_q73_p,cbcl_q81_p,cbcl_q82_p,cbcl_q90_p,cbcl_q96_p,cbcl_q99_p,cbcl_q101_p,cbcl_q105_p,cbcl_q106_p), as.numeric)))	
cbcl_scr_syn_social_r=alpha(data.frame(lapply(select(data,cbcl_q11_p,cbcl_q12_p,cbcl_q25_p,cbcl_q27_p,cbcl_q34_p,cbcl_q36_p,cbcl_q38_p,cbcl_q48_p,cbcl_q62_p,cbcl_q64_p,cbcl_q79_p), as.numeric)))			
cbcl_scr_syn_somatic_r=alpha(data.frame(lapply(select(data,cbcl_q47_p,cbcl_q49_p,cbcl_q51_p,cbcl_q54_p,cbcl_q56a_p,cbcl_q56b_p,cbcl_q56c_p,cbcl_q56d_p,cbcl_q56e_p,cbcl_q56f_p,cbcl_q56g_p), as.numeric)))			
cbcl_scr_syn_thought_r=alpha(data.frame(lapply(select(data,cbcl_q09_p,cbcl_q18_p,cbcl_q40_p,cbcl_q46_p,cbcl_q58_p,cbcl_q59_p,cbcl_q60_p,cbcl_q66_p,cbcl_q70_p,cbcl_q76_p,cbcl_q83_p,cbcl_q84_p,cbcl_q85_p,cbcl_q92_p,cbcl_q100_p), as.numeric)))		
cbcl_scr_syn_withdep_r=alpha(data.frame(lapply(select(data,cbcl_q05_p,cbcl_q42_p,cbcl_q65_p,cbcl_q69_p,cbcl_q75_p,cbcl_q102_p,cbcl_q103_p,cbcl_q111_p), as.numeric)))
cas[1,34]=cbcl_scr_07_ocd_r$total$raw_alpha
cas[1,35]=cbcl_scr_07_sct_r$total$raw_alpha
cas[1,36]=cbcl_scr_07_stress_r$total$raw_alpha
cas[1,37]=cbcl_scr_syn_aggressive_r$total$raw_alpha
cas[1,38]=cbcl_scr_syn_anxdep_r$total$raw_alpha
cas[1,39]=cbcl_scr_syn_withdep_r$total$raw_alpha
cas[1,40]=cbcl_scr_syn_attention_r$total$raw_alpha
cas[1,41]=cbcl_scr_syn_rulebreak_r$total$raw_alpha
cas[1,42]=cbcl_scr_syn_social_r$total$raw_alpha
cas[1,43]=cbcl_scr_syn_somatic_r$total$raw_alpha
cas[1,44]=cbcl_scr_syn_thought_r$total$raw_alpha
colnames(cas)[34]='cbcl_scr_07_ocd_r'
colnames(cas)[35]='cbcl_scr_07_sct_r'
colnames(cas)[36]='cbcl_scr_07_stress_r'
colnames(cas)[37]='cbcl_scr_syn_aggressive_r'
colnames(cas)[38]='cbcl_scr_syn_anxdep_r'
colnames(cas)[39]='cbcl_scr_syn_withdep_r'
colnames(cas)[40]='cbcl_scr_syn_attention_r'
colnames(cas)[41]='cbcl_scr_syn_rulebreak_r'
colnames(cas)[42]='cbcl_scr_syn_social_r'
colnames(cas)[43]='cbcl_scr_syn_somatic_r'
colnames(cas)[44]='cbcl_scr_syn_thought_r'

asr_scr_aggressive_r=alpha(data.frame(lapply(select(data,asr_q03_p,asr_q05_p,asr_q16_p,asr_q28_p,asr_q37_p,asr_q55_p,asr_q57_p,asr_q68_p,asr_q81_p,asr_q86_p,asr_q87_p,asr_q95_p,asr_q114_p,asr_q117_p,asr_q122_p), as.numeric)))	
asr_scr_anxdep_r=alpha(data.frame(lapply(select(data,asr_q12_p,asr_q13_p,asr_q14_p,asr_q22_p,asr_q31_p,asr_q33_p,asr_q34_p,asr_q35_p,asr_q45_p,asr_q47_p,asr_q50_p,asr_q52_p,asr_q71_p,asr_q91_p,asr_q103_p,asr_q107_p,asr_q112_p,asr_q113_p), as.numeric)))	
asr_scr_attention_r=alpha(data.frame(lapply(select(data,asr_q01_p,asr_q08_p,asr_q11_p,asr_q17_p,asr_q53_p,asr_q59_p,asr_q61_p,asr_q64_p,asr_q78_p,asr_q101_p,asr_q102_p,asr_q105_p,asr_q108_p,asr_q119_p,asr_q121_p), as.numeric)))		
asr_scr_intrusive_r=alpha(data.frame(lapply(select(data,asr_q19_p,asr_q74_p,asr_q93_p,asr_q94_p,asr_q104_p), as.numeric)))	
asr_scr_perstr_r=alpha(data.frame(lapply(select(data,asr_q02_p,asr_q04_p,asr_q15_p,asr_q49_p,asr_q73_p,asr_q80_p,asr_q88_p,asr_q98_p,asr_q106_p,asr_q109_p,asr_q123_p), as.numeric)))			
asr_scr_rulebreak_r=alpha(data.frame(lapply(select(data,asr_q06_p,asr_q20_p,asr_q23_p,asr_q26_p,asr_q39_p,asr_q41_p,asr_q43_p,asr_q76_p,asr_q82_p,asr_q90_p,asr_q92_p,asr_q114_p,asr_q117_p,asr_q122_p), as.numeric)))	
asr_scr_somatic_r=alpha(data.frame(lapply(select(data,asr_q51_p,asr_q54_p,asr_q56a_p,asr_q56b_p,asr_q56c_p,asr_q56d_p,asr_q56e_p,asr_q56f_p,asr_q56g_p,asr_q56h_p,asr_q56i_p,asr_q100_p), as.numeric)))		
asr_scr_thought_r=alpha(data.frame(lapply(select(data,asr_q09_p,asr_q18_p,asr_q36_p,asr_q40_p,asr_q46_p,asr_q63_p,asr_q66_p,asr_q70_p,asr_q84_p,asr_q85_p), as.numeric)))		
asr_scr_withdrawn_r=alpha(data.frame(lapply(select(data,asr_q25_p,asr_q30_p,asr_q42_p,asr_q48_p,asr_q60_p,asr_q65_p,asr_q67_p,asr_q69_p,asr_q111_p), as.numeric)))
cas[1,45]=asr_scr_perstr_r$total$raw_alpha
cas[1,46]=asr_scr_aggressive_r$total$raw_alpha
cas[1,47]=asr_scr_anxdep_r$total$raw_alpha
cas[1,48]=asr_scr_attention_r$total$raw_alpha
cas[1,49]=asr_scr_intrusive_r$total$raw_alpha
cas[1,50]=asr_scr_rulebreak_r$total$raw_alpha
cas[1,51]=asr_scr_somatic_r$total$raw_alpha
cas[1,52]=asr_scr_thought_r$total$raw_alpha
cas[1,53]=asr_scr_withdrawn_r$total$raw_alpha
colnames(cas)[45]='asr_scr_perstr_r'
colnames(cas)[46]='asr_scr_aggressive_r'
colnames(cas)[47]='asr_scr_anxdep_r'
colnames(cas)[48]='asr_scr_attention_r'
colnames(cas)[49]='asr_scr_intrusive_r'
colnames(cas)[50]='asr_scr_rulebreak_r'
colnames(cas)[51]='asr_scr_somatic_r'
colnames(cas)[52]='asr_scr_thought_r'
colnames(cas)[53]='asr_scr_withdrawn_r'

library(gdata)
keep(data,backup_data,cas,sure=TRUE)

write.csv(cas,"/home/max/Documents/DRD/es_alphas.csv",row.names = FALSE)

