#!/usr/bin/env python
# coding: utf-8

# ## Load in Packages

# In[1]:


#load packages
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy
from pandas import DataFrame
import csv
import scipy.cluster.hierarchy as shc
import pingouin
import statsmodels.api
import statsmodels.api as sm


# ## Load In Data & Set Variables

# In[2]:


#load in variables and lists needed
file=pd.read_csv('es_data.csv')
data=file
with open('es_varlist_622.txt') as f:
    vars = f.read().splitlines()
proc_vars=['src_subject_id','eventname',"rel_family_id","mri_info_device.serial.number"]
allvars=proc_vars+vars
sid=['src_subject_id']
alphas=pd.read_csv('es_alphas.csv')
alphas=alphas.transpose()
alphas['var1']=alphas.index
alphas['var2']=alphas.index
scanner = data['mri_info_device.serial.number']
#remove siblings so that only 1 sibling remains per family
#data.drop_duplicates(subset ="rel_family_id", 
                     #keep = "first", inplace = True)
for v in proc_vars:
    data=data.drop([v],axis=1)


# ## Filter Data

# In[3]:


#data processing
activities_vars=['sports_activity_activities_p___0','sports_activity_activities_p___1',
                 'sports_activity_activities_p___2','sports_activity_activities_p___3',
                 'sports_activity_activities_p___4','sports_activity_activities_p___5',
                 'sports_activity_activities_p___6','sports_activity_activities_p___7',
                 'sports_activity_activities_p___8','sports_activity_activities_p___9',
                 'sports_activity_activities_p___10','sports_activity_activities_p___11',
                 'sports_activity_activities_p___12','sports_activity_activities_p___13',
                 'sports_activity_activities_p___14','sports_activity_activities_p___15',
                 'sports_activity_activities_p___16','sports_activity_activities_p___17',
                 'sports_activity_activities_p___18','sports_activity_activities_p___19',
                 'sports_activity_activities_p___20','sports_activity_activities_p___21',
                 'sports_activity_activities_p___22','sports_activity_activities_p___23',
                 'sports_activity_activities_p___24','sports_activity_activities_p___25',
                 'sports_activity_activities_p___26','sports_activity_activities_p___27',
                 'sports_activity_activities_p___28']

team_sport_vars=['sports_activity_activities_p___1','sports_activity_activities_p___2',
            'sports_activity_activities_p___4','sports_activity_activities_p___5',
            'sports_activity_activities_p___7','sports_activity_activities_p___11',
            'sports_activity_activities_p___12','sports_activity_activities_p___15',
            'sports_activity_activities_p___21']
ind_sport_vars=['sports_activity_activities_p___3',
           'sports_activity_activities_p___6','sports_activity_activities_p___8',
           'sports_activity_activities_p___9','sports_activity_activities_p___10',
           'sports_activity_activities_p___13','sports_activity_activities_p___14',
           'sports_activity_activities_p___16','sports_activity_activities_p___17',
            'sports_activity_activities_p___18','sports_activity_activities_p___19',
            'sports_activity_activities_p___20','sports_activity_activities_p___22',
           'sports_activity_activities_p___27']
performance_vars=['sports_activity_activities_p___0','sports_activity_activities_p___23',
                  'sports_activity_activities_p___24','sports_activity_activities_p___25']
hobbies_vars=['sports_activity_activities_p___26','sports_activity_activities_p___28']

#set no to 0 and yes 1 (instead of no=1 and yes=2)
for v in activities_vars:
    data[v]=data[v]-1
    
data['sports_activity_activities_p_team_sport']=data['sports_activity_activities_p___1']
for v in team_sport_vars[1:len(team_sport_vars)]:
    data['sports_activity_activities_p_team_sport']=data['sports_activity_activities_p_team_sport']+data[v]
data['sports_activity_activities_p_ind_sport']=data['sports_activity_activities_p___3']
for v in ind_sport_vars[1:len(ind_sport_vars)]:
    data['sports_activity_activities_p_ind_sport']=data['sports_activity_activities_p_ind_sport']+data[v]
data['sports_activity_activities_p_performance']=data['sports_activity_activities_p___0']
for v in performance_vars[1:len(performance_vars)]:
    data['sports_activity_activities_p_performance']=data['sports_activity_activities_p_performance']+data[v]
data['sports_activity_activities_p_hobbies']=data['sports_activity_activities_p___26']
for v in hobbies_vars[1:len(hobbies_vars)]:
    data['sports_activity_activities_p_hobbies']=data['sports_activity_activities_p_hobbies']+data[v]

data['sports_activity_activities_p_team_sport']=data['sports_activity_activities_p_team_sport'].clip(upper=1)
data['sports_activity_activities_p_ind_sport']=data['sports_activity_activities_p_ind_sport'].clip(upper=1)
data['sports_activity_activities_p_performance']=data['sports_activity_activities_p_performance'].clip(upper=1)
data['sports_activity_activities_p_hobbies']=data['sports_activity_activities_p_hobbies'].clip(upper=1)
    

for v in activities_vars:
    data=data.drop([v],axis=1)


# In[4]:


#more data processing
#2
data['su_tlfb_cal_scr_num_events']=data['su_tlfb_cal_scr_num_events'].fillna(0)
#3
data['ksads_back_trans_prob']=data['ksads_back_trans_prob'].fillna(1)
data['ksads_back_sex_orient_probs']=data['ksads_back_sex_orient_probs'].fillna(1)
data['ksads_back_c_trans_prob_p']=data['ksads_back_c_trans_prob_p'].fillna(1)
data['ksads_back_c_gay_prob_p']=data['ksads_back_c_gay_prob_p'].fillna(1)

#9
data['accult_phenx_q4_p']=data['accult_phenx_q4_p'].fillna(1)
data['accult_phenx_q5_p']=data['accult_phenx_q5_p'].fillna(1)
#10

#binarize true categorical demographics to their most common category yes/no
data['race.6level']=data['race.6level'].clip(upper=2)
data['demo_prnt_empl_p']=data['demo_prnt_empl_p'].clip(upper=2)
data['demo_prtnr_empl_p']=data['demo_prtnr_empl_p'].clip(upper=2)
data['demo_prnt_empl_time_p']=data['demo_prnt_empl_time_p'].clip(upper=2)
data['demo_prtnr_empl_time_p']=data['demo_prtnr_empl_time_p'].clip(upper=2)
data['demo_prnt_empl_time_p']=data['demo_prnt_empl_time_p'].clip(upper=2)
#data['demo_prnt_marital_p']=data['demo_prnt_marital_p'].clip(upper=2)

#make aggregate puberty scale
data['pubertdev_ss_female_category']=data['pubertdev_ss_female_category'].fillna(0)
data['pubertdev_ss_male_category']=data['pubertdev_ss_male_category'].fillna(0)
data['puberty']=data['pubertdev_ss_male_category']+data['pubertdev_ss_female_category']
data=data.drop(['pubertdev_ss_male_category'],axis=1)
data=data.drop(['pubertdev_ss_female_category'],axis=1)

#redo ksads var coding with clipping
matching = [s for s in vars if "ksads_" in s]
matching2=[x for x in matching if "_back_" not in x]
matching3=[x for x in matching2 if "_ptsd_" not in x]
for m in matching3:
    data[m]=data[m].clip(upper=1)


# In[5]:


#medhx converted in R: 1  2  4  3 NA  5  6 = 0 1 time 3-4 times  2 times    <NA>  5-9 times  >10+ times
#make ER composite
data['medhx_er_composite']=data['medhx_ss_5b_times_er_before_past_yr_p']+data['medhx_ss_4b_times_er_past_yr_p']
data=data.drop('medhx_ss_5b_times_er_before_past_yr_p',axis=1)
data=data.drop('medhx_ss_4b_times_er_past_yr_p',axis=1)

#correctly coded by R - no change needed
data['hisp']=data['hisp']
data['sex']=data['sex']
data['high.educ.bl']=data['high.educ.bl']
data['household.income.bl']=data['household.income.bl']

#turn all na, refuse, and don't know into nos (no=1)
data['famhx_7_yes_no_p']=data['famhx_7_yes_no_p'].replace(3,1)
data['famhx_8_yes_no_p']=data['famhx_8_yes_no_p'].replace(3,1)
data['famhx_9_yes_no_p']=data['famhx_9_yes_no_p'].replace(3,1)
data['famhx_10_yes_no_p']=data['famhx_10_yes_no_p'].replace(3,1)
data['famhx_11_yes_no_p']=data['famhx_11_yes_no_p'].replace(3,1)
data['famhx_12_yes_no_p']=data['famhx_12_yes_no_p'].replace(3,1)
data['famhx_13_yes_no_p']=data['famhx_13_yes_no_p'].replace(3,1)

data['famhx_7_yes_no_p']=data['famhx_7_yes_no_p'].replace(4,1)
data['famhx_8_yes_no_p']=data['famhx_8_yes_no_p'].replace(4,1)
data['famhx_9_yes_no_p']=data['famhx_9_yes_no_p'].replace(4,1)
data['famhx_10_yes_no_p']=data['famhx_10_yes_no_p'].replace(4,1)
data['famhx_11_yes_no_p']=data['famhx_11_yes_no_p'].replace(4,1)
data['famhx_12_yes_no_p']=data['famhx_12_yes_no_p'].replace(4,1)
data['famhx_13_yes_no_p']=data['famhx_13_yes_no_p'].replace(4,1)

#11
data['devhx_23b_age_wet_bed_p']=data['devhx_23b_age_wet_bed_p'].fillna("Don't know")
#12
data['via_accult_ss_amer_p']=data['via_accult_ss_amer_p'].fillna(8)
data['via_accult_ss_hc_p']=data['via_accult_ss_hc_p'].fillna(0)

#sum of mother problems during birth vars
devhx_vars=['devhx_10a_severe_nausea_p','devhx_10b_heavy_bleeding_p',
            'devhx_10c_eclampsia_p','devhx_10e_persist_proteinuria_p','devhx_10d_gall_bladder_p',
           'devhx_10f_rubella_p','devhx_10g_severe_anemia_p','devhx_10h_urinary_infections_p',
            'devhx_10i_diabetes_p','devhx_10j_high_blood_press_p','devhx_10k_problems_placenta_p',
           'devhx_10l_accident_injury_p','devhx_10m_other_p']
for m in devhx_vars:
    data[m]=data[m].replace("No",0)
for m in devhx_vars:
    data[m]=data[m].replace("Don't know",0)
for m in devhx_vars:
    data[m]=data[m].replace("Yes",1)
data['devhx_mother_probs']=data['devhx_10a_severe_nausea_p']
for v in devhx_vars[1:len(devhx_vars)]:
    data['devhx_mother_probs']= data['devhx_mother_probs']+data[v]
data['devhx_mother_probs'] = data['devhx_mother_probs'].clip(upper=1)
for v in devhx_vars:
    data=data.drop([v],axis=1)
    
#sum of distress of birth vars
devhx_vars=['devhx_14a_blue_birth_p','devhx_14b_slow_heart_beat_p','devhx_14c_did_not_breathe_p',
'devhx_14d_convulsions_p','devhx_14e_jaundice_p','devhx_14f_oxygen_p',
'devhx_14g_blood_transfuse_p','devhx_14h_rh_incompatible_p']

for m in devhx_vars:
    data[m]=data[m].replace("No",0)
for m in devhx_vars:
    data[m]=data[m].replace("Don't know",0)
for m in devhx_vars:
    data[m]=data[m].replace("Yes",1)
data['devhx_distress_at_birth']=data['devhx_14a_blue_birth_p']
for v in devhx_vars[1:len(devhx_vars)]:
    data['devhx_distress_at_birth']= data['devhx_distress_at_birth']+data[v]
data['devhx_distress_at_birth'] = data['devhx_distress_at_birth'].clip(upper=1)
for v in devhx_vars:
    data=data.drop([v],axis=1)
    
#aggregation of milestone meeting
#have added up months at which each occured as a continuous measure of development speed
#roll over late at 6 months, sit CDC Average = 9 months, walk CDC Average = 18 months,
#firstword CDC average = 12
devhx_vars=['devhx_19a_mnths_roll_over_p','devhx_19b_mnths_sit_p',
            'devhx_19c_mnths_walk_p','devhx_19d_first_word_p']
for m in devhx_vars:
    data[m]=data[m].replace("No",0)
for m in devhx_vars:
    data[m]=data[m].replace("Don't know",0)
for m in devhx_vars:
    data[m]=data[m].replace("Yes",1)
data['devhx_milestones']=data['devhx_19a_mnths_roll_over_p']
for v in devhx_vars[1:len(devhx_vars)]:
    data['devhx_milestones']=data['devhx_milestones']+data[v]
for v in devhx_vars:
    data=data.drop([v],axis=1)
    
#set other devhx variables to numeric
devhx_vars=['devhx_10_p','devhx_13_ceasarian_p','devhx_20_motor_dev_p',
           'devhx_21_speech_dev_p','devhx_23b_age_wet_bed_p',
           'devhx_6_pregnancy_planned_p','devhx_caffeine_11_p',
            'devhx_12a_born_premature_p']
for m in devhx_vars:
    data[m]=data[m].replace("No",0)
for m in devhx_vars:
    data[m]=data[m].replace("Don't know",np.nan)
for m in devhx_vars:
    data[m]=data[m].replace("Yes",1)
for m in devhx_vars:
    data[m]=data[m].replace("Yes - at least once a day",1)
    
data=data.drop('devhx_ss_8_her_morph_amt_p',axis=1)
      
#sum up number of friends
data['num_friends']=data['resiliency_5a']+data['resiliency_5b']+data['resiliency_6a']+data['resiliency_6b']
data['male_friends']=data['resiliency_5a']+data['resiliency_5b']
data['female_friends']=data['resiliency_6a']+data['resiliency_6b']
data=data.drop(['resiliency_5a'],axis=1)
data=data.drop(['resiliency_5b'],axis=1)
data=data.drop(['resiliency_6a'],axis=1)
data=data.drop(['resiliency_6b'],axis=1)
data['same_sex_friends'] = np.where(data['sex']=='M', data['male_friends'], data['female_friends'])
data['opposite_sex_friends'] = np.where(data['sex']=='M', data['female_friends'], data['male_friends'])
data=data.drop(['male_friends'],axis=1)
data=data.drop(['female_friends'],axis=1)
data['resiliency_num_friends_cat']=pd.qcut(data['num_friends'],q=5,labels=[1,2,3,4,5])
data['resiliency_same_sex_friends_cat']=pd.qcut(data['same_sex_friends'],q=5,labels=[1,2,3,4,5])
data['resiliency_opposite_sex_friends_cat']=pd.qcut(data['opposite_sex_friends'],q=5,labels=[1,2,3,4,5])
data=data.drop('num_friends',axis=1)
data=data.drop('same_sex_friends',axis=1)
data=data.drop('opposite_sex_friends',axis=1)

#make trauma composite
tra_vars=['ksads_ptsd_raw_754_p','ksads_ptsd_raw_755_p','ksads_ptsd_raw_756_p','ksads_ptsd_raw_757_p',
'ksads_ptsd_raw_758_p','ksads_ptsd_raw_759_p','ksads_ptsd_raw_760_p','ksads_ptsd_raw_761_p',
'ksads_ptsd_raw_762_p','ksads_ptsd_raw_763_p','ksads_ptsd_raw_764_p','ksads_ptsd_raw_765_p',
'ksads_ptsd_raw_766_p','ksads_ptsd_raw_767_p','ksads_ptsd_raw_768_p','ksads_ptsd_raw_769_p',
'ksads_ptsd_raw_770_p']
for m in tra_vars:
    data[m]=data[m].replace("No",0)
for m in tra_vars:
    data[m]=data[m].replace("Don't know",0)
for m in tra_vars:
    data[m]=data[m].replace("Yes",1)
data['ksads_ptsd_composite']=data['ksads_ptsd_raw_754_p']
for v in tra_vars[1:len(tra_vars)]:
    data['ksads_ptsd_composite']=data['ksads_ptsd_composite']+data[v]
data['ksads_ptsd_composite']=(data['ksads_ptsd_composite']-17)
for v in tra_vars:
    data=data.drop([v],axis=1)
data['ksads_ptsd_composite'] = data['ksads_ptsd_composite'].clip(upper=1)

#binarize brain injury
data['brain_injury_ss_agefirst_p']=data['brain_injury_ss_agefirst_p'].fillna(-1)
data['brain_injury_ss_agefirst_p'] = data['brain_injury_ss_agefirst_p'].clip(upper=0)
data['brain_injury_ss_agefirst_p'] = data['brain_injury_ss_agefirst_p'].replace(0,1)
data['brain_injury_ss_agefirst_p'] = data['brain_injury_ss_agefirst_p'].replace(-1,0)

#binarize religion to yes/no
#data['demo_relig_p']=data['demo_relig_p'].replace("Don't know",0)
#data['demo_relig_p']=data['demo_relig_p'].replace("Agnostic (not sure if there is a God)",0)
#data['demo_relig_p']=data['demo_relig_p'].replace("Atheist (do not believe in God)",0)
#data['demo_relig_p']=data['demo_relig_p'].replace("Nothing in Particular",0)
#data['demo_relig_p']=data['demo_relig_p'].replace('Refused to answer',np.nan)
#data['demo_relig_p']=data['demo_relig_p'].replace("Nothing in Particular",0)
#data['demo_relig_p']=data['demo_relig_p'].replace("Evangelical Protestant (e.g., Southern Baptist, Pentecostal, Foursquare Gospel Church, Brethren, Nazarene, Evangelical Christian, Assembly of God)",1)
#ddd=data['demo_relig_p'].unique()
#for d in data['demo_relig_p'].unique():
    #data['demo_relig_p']=data['demo_relig_p'].replace(d,1)
data['demo_relig_p']=data['demo_relig_p'].replace(6,0)
data['demo_relig_p']=data['demo_relig_p'].replace(7,0)
data['demo_relig_p']=data['demo_relig_p'].replace(9,0)
data['demo_relig_p']=data['demo_relig_p'].clip(upper=1)
data['demo_relig_p'].unique()


# In[6]:


#make broader ksads scales
data['ksads_depressive_comp']=(data['ksads_1_840_p']+data['ksads_1_841_p']+data['ksads_1_842_p']
+data['ksads_1_843_p']+data['ksads_1_845_p']+data['ksads_1_846_p']+data['ksads_1_847_p']+data['ksads_1_840_t']+data['ksads_1_841_t']+data['ksads_1_842_t']+data['ksads_1_843_t']+
data['ksads_1_844_t']+data['ksads_1_845_t']+data['ksads_1_846_t']+
data['ksads_1_847_t'])

data['ksads_GAD_composite']=(data['ksads_10_869_p']+data['ksads_10_869_t']+data['ksads_10_870_p']+
data['ksads_10_870_t']+data['ksads_10_913_p']+data['ksads_10_913_t']+
data['ksads_10_914_p']+data['ksads_10_914_t'])

data['ksads_OCD_composite']=(data['ksads_11_917_p']+data['ksads_11_918_p']+data['ksads_11_919_p']+data['ksads_11_920_p'])
data['ksads_OCD_composite']=data['ksads_OCD_composite'].clip(upper=1)

data['ksads_eating_disorder_composite']=(data['ksads_13_929_p']+data['ksads_13_930_p']+data['ksads_13_931_p']+
data['ksads_13_932_p']+data['ksads_13_933_p']+data['ksads_13_934_p']+data['ksads_13_935_p']+data['ksads_13_936_p']+
data['ksads_13_937_p']+data['ksads_13_938_p']+data['ksads_13_939_p']+data['ksads_13_940_p']+data['ksads_13_941_p']+
data['ksads_13_942_p']+data['ksads_13_943_p']+data['ksads_13_944_p'])
data['ksads_eating_disorder_composite']=data['ksads_eating_disorder_composite'].clip(upper=1)

data['ksads_adhd_composite']=(data['ksads_14_853_p']+data['ksads_14_854_p']
                              +data['ksads_14_855_p']+data['ksads_14_856_p'])
data['ksads_adhd_composite']=data['ksads_adhd_composite'].clip(upper=1)

data['ksads_cd_composite']=(data['ksads_16_897_p']+data['ksads_16_898_p']+data['ksads_16_899_p']
+data['ksads_16_900_p'])
data['ksads_cd_composite']=data['ksads_cd_composite'].clip(upper=1)

data['ksads_bipolar_composite']=(data['ksads_2_830_p']+
data['ksads_2_830_t']+data['ksads_2_831_p']+data['ksads_2_831_t']+data['ksads_2_832_p']+
data['ksads_2_832_t']+data['ksads_2_833_p']+data['ksads_2_833_t']+data['ksads_2_834_p']+
data['ksads_2_834_t']+data['ksads_2_835_p']+data['ksads_2_835_t']+data['ksads_2_836_p']+
data['ksads_2_836_t']+data['ksads_2_837_p']+data['ksads_2_837_t']+data['ksads_2_838_p']+
data['ksads_2_838_t']+data['ksads_2_839_p']+data['ksads_2_839_t'])


data['ksads_sud_composite']=(data['ksads_20_888_p']+data['ksads_20_889_p']+data['ksads_20_890_p']+
                            data['ksads_20_893_p']+data['ksads_20_894_p'])

data['ksads_nssi_composite']=(data['ksads_23_945_p']+data['ksads_23_945_t']+data['ksads_23_956_p']+
                              data['ksads_23_956_t'])
                             
data['ksads_suicide_composite']=(data['ksads_23_946_p']+data['ksads_23_946_t']+data['ksads_23_947_p']
                            +data['ksads_23_947_t']+data['ksads_23_948_p']+data['ksads_23_948_t']
                             +data['ksads_23_956_p']
                            +data['ksads_23_949_p']+data['ksads_23_946_t']+data['ksads_23_947_p']
                            +data['ksads_23_947_t']+data['ksads_23_948_p']+data['ksads_23_948_t']
                             +data['ksads_23_949_t']+data['ksads_23_950_p']
                              +data['ksads_23_947_t']+data['ksads_23_948_p']+data['ksads_23_948_t']
                             +data['ksads_23_950_t']+data['ksads_23_950_p']
                            +data['ksads_23_951_p']+data['ksads_23_951_t']+data['ksads_23_952_p']
                            +data['ksads_23_952_t']+data['ksads_23_953_p']+data['ksads_23_953_t']
                             +data['ksads_23_954_p']+data['ksads_23_954_t']
                            +data['ksads_23_957_p']+data['ksads_23_957_t']+data['ksads_23_958_p']
                             +data['ksads_23_958_t']
                            +data['ksads_23_959_p']+data['ksads_23_959_t']+data['ksads_23_960_p']
                            +data['ksads_23_960_t']+data['ksads_23_961_p']+data['ksads_23_961_t']
                             +data['ksads_23_962_p']+data['ksads_23_962_t']
                              +data['ksads_23_963_p']+data['ksads_23_963_t']+data['ksads_23_964_p']
                             +data['ksads_23_964_t']+data['ksads_23_965_p']
                            +data['ksads_23_965_t'])

data['ksads_psychosis_composite']=(data['ksads_4_826_p']+data['ksads_4_827_p']
                                 +data['ksads_4_828_p']+data['ksads_4_829_p']+data['ksads_4_849_p']
                                 +data['ksads_4_850_p']+data['ksads_4_851_p']+data['ksads_4_852_p'])
                             
data['ksads_SAD_composite']=(data['ksads_8_863_p']+data['ksads_8_863_t']
                                 +data['ksads_8_864_p']+data['ksads_8_864_t']+data['ksads_8_911_p']
                                 +data['ksads_8_911_t']+data['ksads_8_912_p']+data['ksads_8_912_t'])

data['ksads_SAD_composite']=data['ksads_SAD_composite'].clip(upper=1)
data['ksads_psychosis_composite']=data['ksads_psychosis_composite'].clip(upper=1)
data['ksads_suicide_composite']=data['ksads_suicide_composite'].clip(upper=1)
data['ksads_nssi_composite']=data['ksads_nssi_composite'].clip(upper=1)
data['ksads_sud_composite']=data['ksads_sud_composite'].clip(upper=1)
data['ksads_bipolar_composite']=data['ksads_bipolar_composite'].clip(upper=1)

data['ksads_SAD_composite']=data['ksads_SAD_composite'].astype(float)
data['ksads_psychosis_composite']=data['ksads_psychosis_composite'].astype(float)
data['ksads_suicide_composite']=data['ksads_suicide_composite'].astype(float)
data['ksads_nssi_composite']=data['ksads_nssi_composite'].astype(float)
data['ksads_sud_composite']=data['ksads_sud_composite'].astype(float)
data['ksads_bipolar_composite']=data['ksads_bipolar_composite'].astype(float)

ksads_raw=['ksads_1_840_p','ksads_1_840_t','ksads_1_841_p','ksads_1_841_t','ksads_1_842_p',
'ksads_1_842_t','ksads_1_843_p','ksads_1_843_t','ksads_1_844_p','ksads_1_844_t',
'ksads_1_845_p','ksads_1_845_t','ksads_1_846_p','ksads_1_846_t','ksads_1_847_p','ksads_1_847_t','ksads_10_869_p','ksads_10_869_t','ksads_10_870_p','ksads_10_870_t','ksads_10_913_p',
'ksads_10_913_t','ksads_10_914_p','ksads_10_914_t','ksads_11_917_p','ksads_11_918_p','ksads_11_919_p',
'ksads_11_920_p','ksads_13_929_p','ksads_13_930_p','ksads_13_931_p','ksads_13_932_p','ksads_13_933_p',
'ksads_13_934_p','ksads_13_935_p','ksads_13_936_p','ksads_13_937_p','ksads_13_938_p','ksads_13_939_p',
'ksads_13_940_p','ksads_13_941_p','ksads_13_942_p','ksads_13_943_p',
'ksads_13_944_p','ksads_14_853_p','ksads_14_854_p','ksads_14_855_p',
'ksads_14_856_p','ksads_16_897_p','ksads_16_898_p','ksads_16_899_p',
'ksads_16_900_p','ksads_18_903_p','ksads_2_830_p','ksads_2_830_t',
'ksads_2_831_p','ksads_2_831_t','ksads_2_832_p','ksads_2_832_t','ksads_2_833_p',
'ksads_2_833_t','ksads_2_834_p','ksads_2_834_t','ksads_2_835_p','ksads_2_835_t','ksads_2_836_p','ksads_2_836_t','ksads_2_837_p','ksads_2_837_t','ksads_2_838_p',
'ksads_2_838_t','ksads_2_839_p','ksads_2_839_t','ksads_20_888_p','ksads_20_889_p','ksads_20_890_p','ksads_20_893_p',
'ksads_20_894_p','ksads_23_945_p','ksads_23_945_t','ksads_23_946_p','ksads_23_946_t','ksads_23_947_p','ksads_23_947_t','ksads_23_948_p',
'ksads_23_948_t','ksads_23_949_p','ksads_23_949_t','ksads_23_950_p','ksads_23_950_t','ksads_23_951_p',
'ksads_23_951_t','ksads_23_952_p','ksads_23_952_t','ksads_23_953_p','ksads_23_953_t','ksads_23_954_p',
'ksads_23_954_t','ksads_23_956_p','ksads_23_956_t','ksads_23_957_p','ksads_23_957_t',
'ksads_23_958_p','ksads_23_958_t','ksads_23_959_p','ksads_23_959_t','ksads_23_960_p','ksads_23_960_t',
'ksads_23_961_p','ksads_23_961_t','ksads_23_962_p','ksads_23_962_t','ksads_23_963_p','ksads_23_963_t',
'ksads_23_964_p','ksads_23_964_t','ksads_23_965_p','ksads_23_965_t','ksads_4_826_p','ksads_4_827_p',
'ksads_4_828_p','ksads_4_829_p','ksads_4_849_p','ksads_4_850_p',
'ksads_4_851_p','ksads_4_852_p','ksads_8_863_p','ksads_8_863_t',
'ksads_8_864_p','ksads_8_864_t','ksads_8_911_p','ksads_8_911_t','ksads_8_912_p','ksads_8_912_t']
for v in ksads_raw:
    data=data.drop([v],axis=1)


# In[7]:


#more data processing
data['accult_phenx_q45_p']=data['accult_phenx_q4_p']+data['accult_phenx_q5_p']
data['accult_phenx_q45_p']=data['accult_phenx_q45_p']/2
data=data.drop(['accult_phenx_q4_p'],axis=1)
data=data.drop(['accult_phenx_q5_p'],axis=1)

data['devhx_ss_alcohol_avg_p']=data['devhx_ss_8_alcohol_avg_p']+data['devhx_ss_9_alcohol_avg_p']
data['devhx_ss_alcohol_effects_p']=data['devhx_ss_8_alcohol_effects_p']+data['devhx_ss_9_alcohol_effects_p']
data['devhx_ss_alcohol_max_p']=data['devhx_ss_8_alcohol_max_p']+data['devhx_ss_9_alcohol_max_p']
data['devhx_ss_cigs_per_day_p']=data['devhx_ss_8_cigs_per_day_p']+data['devhx_ss_9_cigs_per_day_p']
data['devhx_ss_coc_crack_amt_p']=data['devhx_ss_8_coc_crack_amt_p']+data['devhx_ss_9_coc_crack_amt_p']
data['devhx_ss_marijuana_amt_p']=data['devhx_ss_8_marijuana_amt_p']+data['devhx_ss_9_marijuana_amt_p']
data['devhx_ss_oxycont_amt_p']=data['devhx_ss_8_oxycont_amt_p']+data['devhx_ss_9_oxycont_amt_p']

data=data.drop(['devhx_ss_8_oxycont_amt_p'],axis=1)
data=data.drop(['devhx_ss_8_marijuana_amt_p'],axis=1)
data=data.drop(['devhx_ss_8_alcohol_avg_p'],axis=1)
data=data.drop(['devhx_ss_8_alcohol_effects_p'],axis=1)
data=data.drop(['devhx_ss_8_alcohol_max_p'],axis=1)
data=data.drop(['devhx_ss_8_coc_crack_amt_p'],axis=1)
data=data.drop(['devhx_ss_8_cigs_per_day_p'],axis=1)

data=data.drop(['devhx_ss_9_oxycont_amt_p'],axis=1)
data=data.drop(['devhx_ss_9_marijuana_amt_p'],axis=1)
data=data.drop(['devhx_ss_9_alcohol_avg_p'],axis=1)
data=data.drop(['devhx_ss_9_alcohol_effects_p'],axis=1)
data=data.drop(['devhx_ss_9_alcohol_max_p'],axis=1)
data=data.drop(['devhx_ss_9_coc_crack_amt_p'],axis=1)
data=data.drop(['devhx_ss_9_cigs_per_day_p'],axis=1)


# In[8]:


#more data processing
data['screentime_week_p']=(data['screentime_1_hours_p']+(data['screentime_1_minutes_p']/60))
data['screentime_weekend_p']=(data['screentime_2_hours_p']+(data['screentime_2_minutes_p']/60))
data=data.drop(['screentime_1_hours_p'],axis=1)
data=data.drop(['screentime_1_minutes_p'],axis=1)
data=data.drop(['screentime_2_hours_p'],axis=1)
data=data.drop(['screentime_2_minutes_p'],axis=1)

data['su_crpf_avail_sum']=(data['su_crpf_avail_1_p']+data['su_crpf_avail_2_p']+data['su_crpf_avail_3_p']+
data['su_crpf_avail_4_p']+data['su_crpf_avail_5_p']+data['su_crpf_avail_6_p'])

data=data.drop(['su_crpf_avail_1_p'],axis=1)
data=data.drop(['su_crpf_avail_2_p'],axis=1)
data=data.drop(['su_crpf_avail_3_p'],axis=1)
data=data.drop(['su_crpf_avail_4_p'],axis=1)
data=data.drop(['su_crpf_avail_5_p'],axis=1)
data=data.drop(['su_crpf_avail_6_p'],axis=1)

data['famhx_total']=data['famhx_7_yes_no_p']+data['famhx_8_yes_no_p']+data['famhx_9_yes_no_p']+data['famhx_10_yes_no_p']+data['famhx_11_yes_no_p']+data['famhx_12_yes_no_p']+data['famhx_13_yes_no_p']

data=data.drop(['famhx_7_yes_no_p'],axis=1)
data=data.drop(['famhx_8_yes_no_p'],axis=1)
data=data.drop(['famhx_9_yes_no_p'],axis=1)
data=data.drop(['famhx_10_yes_no_p'],axis=1)
data=data.drop(['famhx_11_yes_no_p'],axis=1)
data=data.drop(['famhx_12_yes_no_p'],axis=1)
data=data.drop(['famhx_13_yes_no_p'],axis=1)


# In[9]:


#make last changes to edge cases
data['devhx_10_p']=data['devhx_10_p'].replace('-1',np.nan)
data['neurocog_cash_choice_task']=data['neurocog_cash_choice_task'].replace(3,np.nan)
data['demo_prnt_ethn_p']=data['demo_prnt_ethn_p'].replace(3,np.nan)
data['demo_prnt_ethn_p']=data['demo_prnt_ethn_p'].replace(4,np.nan)
data['demo_prnt_gender_id_p']=data['demo_prnt_gender_id_p'].replace(3,1)
data['demo_prnt_gender_id_p']=data['demo_prnt_gender_id_p'].replace(4,2)
for r in range(5,9):
    data['demo_prnt_gender_id_p']=data['demo_prnt_gender_id_p'].replace(r,np.nan)
data['demo_gender_id_p']=data['demo_gender_id_p'].replace(8,1)
data['demo_gender_id_p']=data['demo_gender_id_p'].replace(4,2)
for r in range(3,10):
    data['demo_gender_id_p']=data['demo_gender_id_p'].replace(r,np.nan)
data['devhx_16_days_high_fever_p']=data['devhx_16_days_high_fever_p'].replace(9990,np.nan)
data['devhx_20_motor_dev_p']=data['devhx_20_motor_dev_p'].replace(6,np.nan)
data['devhx_21_speech_dev_p']=data['devhx_21_speech_dev_p'].replace(6,np.nan)
data['devhx_23b_age_wet_bed_p']=data['devhx_23b_age_wet_bed_p'].replace("Don't know",np.nan)
data['devhx_23b_age_wet_bed_p']=data['devhx_23b_age_wet_bed_p'].replace("Still wets bed",10)
data=data.drop(['devhx_2b_birth_wt_oz_p'],axis=1)
data['devhx_5_twin_p']=data['devhx_5_twin_p'].replace(3,np.nan)
data['devhx_6_pregnancy_planned_p']=data['devhx_6_pregnancy_planned_p'].replace(3,np.nan)
data['devhx_12a_born_premature_p']=data['devhx_12a_born_premature_p'].replace(3,np.nan)
data['ehi_ss_score']=data['ehi_ss_score'].replace(3,4)
data['ehi_ss_score']=data['ehi_ss_score'].replace(2,3)
data['ehi_ss_score']=data['ehi_ss_score'].replace(4,2)
data['ksads_back_c_best_friend_p']=data['ksads_back_c_best_friend_p'].replace(1,np.nan)
data['ksads_back_c_best_friend_p']=data['ksads_back_c_best_friend_p'].replace(2,1)
data['ksads_back_c_best_friend_p']=data['ksads_back_c_best_friend_p'].replace(3,2)
data['ksads_back_c_det_susp_p']=data['ksads_back_c_det_susp_p'].replace(3,np.nan)
data['ksads_back_c_det_susp_p']=data['ksads_back_c_det_susp_p'].replace(4,np.nan)
data['ksads_back_c_drop_in_grades_p']=data['ksads_back_c_drop_in_grades_p'].replace(1,np.nan)
data['ksads_back_c_drop_in_grades_p']=data['ksads_back_c_drop_in_grades_p'].replace(2,1)
data['ksads_back_c_drop_in_grades_p']=data['ksads_back_c_drop_in_grades_p'].replace(3,2)
data['ksads_back_c_gay_p']=data['ksads_back_c_gay_p'].replace(4,np.nan)
data['ksads_back_c_gay_prob_p']=data['ksads_back_c_gay_prob_p'].replace(4,np.nan)
data['ksads_back_c_trans_p']=data['ksads_back_c_trans_p'].replace(4,np.nan)
data['ksads_back_c_trans_prob_p']=data['ksads_back_c_trans_prob_p'].replace(4,np.nan)
data['ksads_back_c_how_well_school_p']=data['ksads_back_c_how_well_school_p'].replace(1,np.nan)
data['ksads_back_c_mh_sa_p']=data['ksads_back_c_mh_sa_p'].replace(3,np.nan)
data['ksads_back_c_school_setting_p']=data['ksads_back_c_school_setting_p'].replace(3,1)
data['ksads_back_c_school_setting_p']=data['ksads_back_c_school_setting_p'].replace(4,1)
data['ksads_back_c_school_setting_p']=data['ksads_back_c_school_setting_p'].replace(5,1)
data['ksads_back_c_school_setting_p']=data['ksads_back_c_school_setting_p'].replace(6,1)
data['ksads_back_c_school_setting_p']=data['ksads_back_c_school_setting_p'].replace(7,1)
data['ksads_back_c_school_setting_p']=data['ksads_back_c_school_setting_p'].replace(8,1)
data['ksads_back_c_school_setting_p']=data['ksads_back_c_school_setting_p'].replace(9,1)
data['ksads_back_conflict_p']=data['ksads_back_conflict_p'].replace(1,np.nan)
data['ksads_back_grades_in_school_p']=data['ksads_back_grades_in_school_p'].replace(1,np.nan)
data['ksads_back_grades_in_school_p']=data['ksads_back_grades_in_school_p'].replace(7,np.nan)
data['ksads_back_sex_orient']=data['ksads_back_sex_orient'].replace(4,np.nan)
data['ksads_back_trans_id']=data['ksads_back_trans_id'].replace(4,np.nan)
data['accult_phenx_q1_p']=data['accult_phenx_q1_p'].replace(6,np.nan)
data['accult_phenx_q1_p']=data['accult_phenx_q1_p'].replace(5,np.nan)
data['devhx_15_days_incubator_p']=data['devhx_15_days_incubator_p'].replace(999,np.nan)
data['devhx_4_age_at_birth_father_p']=data['devhx_4_age_at_birth_father_p'].replace(389,np.nan)
data['devhx_4_age_at_birth_father_p']=data['devhx_4_age_at_birth_father_p'].replace(332,np.nan)
data['devhx_caffeine_11_p']=data['devhx_caffeine_11_p'].replace(6,np.nan)
data['devhx_caffeine_11_p']=data['devhx_caffeine_11_p'].replace(1,np.nan)
data['devhx_caffeine_11_p']=data['devhx_caffeine_11_p'].replace(2,1)
data['devhx_caffeine_11_p']=data['devhx_caffeine_11_p'].replace(5,2)
data['devhx_caffeine_11_p']=data['devhx_caffeine_11_p'].replace(4,33)
data['devhx_caffeine_11_p']=data['devhx_caffeine_11_p'].replace(3,4)
data['devhx_caffeine_11_p']=data['devhx_caffeine_11_p'].replace(33,3)


# In[10]:


#drop construct contamination vars
data=data.drop(['asr_scr_external_r'],axis=1)
data=data.drop(['asr_scr_internal_r'],axis=1)
data=data.drop(['asr_scr_adhd_r'],axis=1)
data=data.drop(['asr_scr_antisocial_r'],axis=1)
data=data.drop(['asr_scr_depress_r'],axis=1)
data=data.drop(['asr_scr_hyperactive_r'],axis=1)
data=data.drop(['asr_scr_inattention_r'],axis=1)
data=data.drop(['asr_scr_anxdisord_r'],axis=1)
data=data.drop(['asr_scr_somaticpr_r'],axis=1)
#data=data.drop(['asr_scr_totprob_r'],axis=1)

data=data.drop(['cbcl_scr_syn_internal_r'],axis=1)
data=data.drop(['cbcl_scr_syn_external_r'],axis=1)
data=data.drop(['cbcl_scr_dsm5_adhd_r'],axis=1)
data=data.drop(['cbcl_scr_dsm5_anxdisord_r'],axis=1)
data=data.drop(['cbcl_scr_dsm5_conduct_r'],axis=1)
data=data.drop(['cbcl_scr_dsm5_depress_r'],axis=1)
data=data.drop(['cbcl_scr_dsm5_opposit_r'],axis=1)
data=data.drop(['cbcl_scr_dsm5_somaticpr_r'],axis=1)
#data=data.drop(['cbcl_scr_syn_totprob_r'],axis=1)

data=data.drop(['cbcl_scr_syn_totprob_r','asr_scr_avoidant_r','asr_scr_totprob_r','nihtbx_cryst_uncorrected',
                'nihtbx_fluidcomp_uncorrected','nihtbx_totalcomp_uncorrected','resiliency_num_friends_cat'],axis=1)


# In[11]:


#drop variables very high in missingness
data=data.drop(['bpmt_scr_attention_r', 'bpmt_scr_external_r', 'bpmt_scr_internal_r',
       'devhx_ss_9_her_morph_amt_p', 'ksads_back_c_reg_friend_group_opin_p',
       'ksads_back_c_reg_friend_group_p'],axis=1)


# In[12]:


#print out final dataset and varnames
data.to_csv("dataset_postproc.csv")
cns=data.columns
df = pd.DataFrame(cns)
df.columns=['names']
df.to_csv("col_names.csv")


# # Analyses

# In[13]:


datacr=data.dropna(axis=1,how="all")
datacr=datacr.sort_index(axis=1)
data_fin=datacr
crr=datacr.corr(method='pearson')
crr.to_csv("cormat.csv")
plt.title("Correlation Among All Variables")
plt.xlabel('Variable Names')
plt.ylabel('Variable Names')
sn.heatmap(crr,cmap="RdBu_r")
plt.show()
crr3 = crr.stack().reset_index()
#set column names
crr3.columns = ['var1','var2','cor']
crrb=crr3
crr3=crr3[crr3['cor']<1]
crr3.drop_duplicates(subset ="cor", 
                     keep = "first", inplace = True) 
crr3=crr3.sort_values(by='cor')
crr3.to_csv("stacked_cormat.csv")
plt.hist(crr3['cor'],bins=100)
plt.title("All Variables - pos & neg")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()


# In[14]:


crr=abs(crr)
crr.to_csv("abs_cormat.csv")
sn.heatmap(crr,cmap="Reds")
plt.title("Correlation Among All Variables")
plt.xlabel('Variable Names')
plt.ylabel('Variable Names')
plt.show()
crr4 = crr.stack().reset_index()
#set column names
#crr4['2scor']=crrb['cor']
crr4.columns = ['var1','var2','cor']
crr4=crr4[crr4['cor']<1]
crr4.drop_duplicates(subset ="cor", 
                     keep = "first", inplace = True) 
crr4=crr4.sort_values(by='cor',ascending=False).reset_index()
crr4.to_csv("stacked_cormat_abs_pd.csv")
plt.hist(crr4['cor'],bins=100)
plt.title("All Variables")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.savefig('Figure_1.tiff', dpi=300)
plt.show()
crr_fin=crr4
#Quantiles
agg_quart=DataFrame()
agg_quart['all']=crr4['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
crr4['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])


# In[15]:


#Calculate between instrument correlations
cr=crr4
cr['redun'] = cr['var1'].astype(str).str[0:3]==cr['var2'].astype(str).str[0:3]
zscore=cr
cr=cr[cr['redun']==True]
plt.hist(cr['cor'],bins=100)
plt.title("Within Instrument Correlations")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
agg_quart['within-instrument']=cr['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
cr.quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95],axis=0)


# In[16]:


#Calculate Between Instrument Correlations
cr=crr4
cr['redun'] = cr['var1'].astype(str).str[0:3]==cr['var2'].astype(str).str[0:3]
cr=cr[cr['redun']==False]
plt.hist(cr['cor'],bins=100)
plt.title("Between Instrument Correlations")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
#cr.to_csv("noredun_stacked_cormat_abs.csv")
agg_quart['between-instrument_correct']=cr['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
cr.quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95],axis=0)


# In[17]:


#read in list of variable domain names
col_groups=pd.read_csv('col_groups5.csv')
col_groups_domain=col_groups
var_ls_mental_health=col_groups['mental_health'].values.tolist()
var_ls_sociodemographic=col_groups['sociodemographic'].values.tolist()
var_ls_biological=col_groups['biological'].values.tolist()
var_ls_cognitive=col_groups['cognitive'].values.tolist()
var_ls_personality=col_groups['personality/trait'].values.tolist()
var_ls_family=col_groups['social/family/environment'].values.tolist()
var_ls_sociodemographic = [x for x in var_ls_sociodemographic if str(x) != 'nan']
var_ls_biological=[x for x in var_ls_biological if str(x) != 'nan']
var_ls_cognitive=[x for x in var_ls_cognitive if str(x) != 'nan']
var_ls_personality=[x for x in var_ls_personality if str(x) != 'nan']
var_ls_family=[x for x in var_ls_family if str(x) != 'nan']
data_mentalhealth=data[var_ls_mental_health]
data_sociodemographic=data[var_ls_sociodemographic]
data_biological=data[var_ls_biological]
data_cognitive=data[var_ls_cognitive]
data_personality=data[var_ls_personality]
data_family=data[var_ls_family]
var_ls_psych=var_ls_cognitive+var_ls_personality
var_ls_context=var_ls_sociodemographic+var_ls_family


# In[18]:


#read in lists of variable informant/self classes
col_groups=pd.read_csv('classes_manual_as_colgroups5.csv')
child_parentreport=col_groups['Child_parent-report'].values.tolist()
child_selfreport=col_groups['Child_Self-report'].values.tolist()
child_task=col_groups['Child_Task'].values.tolist()
family_parentreport=col_groups['Family_parent-report'].values.tolist()
parent_parentreport=col_groups['Parent_parent-report'].values.tolist()
child_parentreport = [x for x in child_parentreport if str(x) != 'nan']
child_selfreport=[x for x in child_selfreport if str(x) != 'nan']
child_task=[x for x in child_task if str(x) != 'nan']
family_parentreport=[x for x in family_parentreport if str(x) != 'nan']
parent_parentreport=[x for x in parent_parentreport if str(x) != 'nan']
data_child_parentreport=data[child_parentreport]
data_child_selfreport=data[child_selfreport]
data_child_task=data[child_task]
data_family_parentreport=data[family_parentreport]
data_parent_parentreport=data[parent_parentreport]


# In[19]:


#make cross-correlation matrices (inter-reporter & inter-domain)
cpr_csr=pd.concat([data_child_parentreport, data_child_selfreport], axis=1, keys=['data_child_parentreport', 'data_child_selfreport']).corr().loc['data_child_selfreport', 'data_child_parentreport']
cpr_ct=pd.concat([data_child_parentreport, data_child_task], axis=1, keys=['data_child_parentreport', 'data_child_task']).corr().loc['data_child_task', 'data_child_parentreport']
cpr_fpr=pd.concat([data_child_parentreport, data_family_parentreport], axis=1, keys=['data_child_parentreport', 'data_family_parentreport']).corr().loc['data_family_parentreport', 'data_child_parentreport']
cpr_ppr=pd.concat([data_child_parentreport, data_parent_parentreport], axis=1, keys=['data_child_parentreport', 'data_parent_parentreport']).corr().loc['data_parent_parentreport', 'data_child_parentreport']
csr_ct=pd.concat([data_child_selfreport, data_child_task], axis=1, keys=['data_child_selfreport', 'data_child_task']).corr().loc['data_child_task', 'data_child_selfreport']
csr_fpr=pd.concat([data_child_selfreport, data_family_parentreport], axis=1, keys=['data_child_selfreport', 'data_family_parentreport']).corr().loc['data_family_parentreport', 'data_child_selfreport']
csr_ppr=pd.concat([data_child_selfreport, data_parent_parentreport], axis=1, keys=['data_child_selfreport', 'data_parent_parentreport']).corr().loc['data_parent_parentreport', 'data_child_selfreport']
ct_fpr=pd.concat([data_child_task, data_family_parentreport], axis=1, keys=['data_child_task', 'data_family_parentreport']).corr().loc['data_family_parentreport', 'data_child_task']
ct_ppr=pd.concat([data_child_task, data_parent_parentreport], axis=1, keys=['data_child_task', 'data_parent_parentreport']).corr().loc['data_parent_parentreport', 'data_child_task']
fpr_ppr=pd.concat([data_family_parentreport, data_parent_parentreport], axis=1, keys=['data_child_task', 'data_parent_parentreport']).corr().loc['data_parent_parentreport', 'data_child_task']

mh_sd=pd.concat([data_mentalhealth, data_sociodemographic], axis=1, keys=['data_mentalhealth', 'data_sociodemographic']).corr().loc['data_sociodemographic', 'data_mentalhealth']
mh_bio=pd.concat([data_mentalhealth, data_biological], axis=1, keys=['data_mentalhealth', 'data_biological']).corr().loc['data_biological', 'data_mentalhealth']
mh_cog=pd.concat([data_mentalhealth, data_cognitive], axis=1, keys=['data_mentalhealth', 'data_cognitive']).corr().loc['data_cognitive', 'data_mentalhealth']
mh_per=pd.concat([data_mentalhealth, data_personality], axis=1, keys=['data_mentalhealth', 'data_personality']).corr().loc['data_personality', 'data_mentalhealth']
mh_fam=pd.concat([data_mentalhealth, data_family], axis=1, keys=['data_mentalhealth', 'data_family']).corr().loc['data_family', 'data_mentalhealth']
sd_bio=pd.concat([data_sociodemographic, data_biological], axis=1, keys=['data_sociodemographic', 'data_biological']).corr().loc['data_biological', 'data_sociodemographic']
sd_cog=pd.concat([data_sociodemographic, data_cognitive], axis=1, keys=['data_sociodemographic', 'data_cognitive']).corr().loc['data_cognitive', 'data_sociodemographic']
sd_per=pd.concat([data_sociodemographic, data_personality], axis=1, keys=['data_sociodemographic', 'data_personality']).corr().loc['data_personality', 'data_sociodemographic']
sd_fam=pd.concat([data_sociodemographic, data_family], axis=1, keys=['data_sociodemographic', 'data_family']).corr().loc['data_family', 'data_sociodemographic']
bio_cog=pd.concat([data_biological, data_cognitive], axis=1, keys=['data_biological', 'data_cognitive']).corr().loc['data_cognitive', 'data_biological']
bio_per=pd.concat([data_biological, data_personality], axis=1, keys=['data_biological', 'data_personality']).corr().loc['data_personality', 'data_biological']
bio_bam=pd.concat([data_biological, data_family], axis=1, keys=['data_biological', 'data_family']).corr().loc['data_family', 'data_biological']
cog_per=pd.concat([data_cognitive, data_personality], axis=1, keys=['data_cognitive', 'data_personality']).corr().loc['data_personality', 'data_cognitive']
cog_fam=pd.concat([data_cognitive, data_family], axis=1, keys=['data_cognitive', 'data_family']).corr().loc['data_family', 'data_cognitive']
per_fam=pd.concat([data_personality, data_family], axis=1, keys=['data_personality', 'data_family']).corr().loc['data_family', 'data_personality']


# In[20]:


#build within/between domain variable for supplemental analysis in R
domains=['mental_health','sociodemographic','biological','cognitive','personality/trait','social/family/environment']
           
#str(reporters[1])
for r in range(len(domains)):
    zscore[str(domains[r]+'_var1')]=0
    zscore[str(domains[r]+'_var2')]=0
    i=0
    for v in zscore['var1']:
        if v in list(col_groups_domain[str(domains[r])]):
            zscore[str(domains[r]+'_var1')][i]=1
        else:
            zscore[str(domains[r]+'_var1')][i]=0
        i+=1
    i=0
    for v in zscore['var2']:
        if v in list(col_groups_domain[str(domains[r])]):
            zscore[str(domains[r]+'_var2')][i]=1
        else:
            zscore[str(domains[r]+'_var2')][i]=0
        i+=1

zscore['within_domain']=0
zscore['between_domain']=0

zscore['within_domain']=zscore['within_domain']+(zscore['mental_health_var1'] & zscore['mental_health_var2'])
zscore['within_domain']=zscore['within_domain']+(zscore['sociodemographic_var1'] & zscore['sociodemographic_var2'])
zscore['within_domain']=zscore['within_domain']+(zscore['biological_var1'] & zscore['biological_var2'])
zscore['within_domain']=zscore['within_domain']+(zscore['cognitive_var1'] & zscore['cognitive_var2'])
zscore['within_domain']=zscore['within_domain']+(zscore['personality/trait_var1'] & zscore['personality/trait_var2'])
zscore['within_domain']=zscore['within_domain']+(zscore['social/family/environment_var1'] & zscore['social/family/environment_var2'])

zscore['between_domain']=abs(zscore['within_domain']-1)


# In[21]:


#build within/between reporter variable for supplemental analysis in R
reporters=['Child_Factual','Child_parent-report','Child_Self-report','Child_Task','Family_parent-report','Parent_parent-report']
           
#str(reporters[1])
for r in range(len(reporters)):
    zscore[str(reporters[r]+'_var1')]=0
    zscore[str(reporters[r]+'_var2')]=0
    i=0
    for v in zscore['var1']:
        if v in list(col_groups[str(reporters[r])]):
            zscore[str(reporters[r]+'_var1')][i]=1
        else:
            zscore[str(reporters[r]+'_var1')][i]=0
        i+=1
    i=0
    for v in zscore['var2']:
        if v in list(col_groups[str(reporters[r])]):
            zscore[str(reporters[r]+'_var2')][i]=1
        else:
            zscore[str(reporters[r]+'_var2')][i]=0
        i+=1

zscore['within_reporter']=0
zscore['between_reporter']=0

zscore['within_reporter']=zscore['within_reporter']+(zscore['Child_Task_var1'] & zscore['Child_Task_var2'])
zscore['within_reporter']=zscore['within_reporter']+(zscore['Child_Factual_var1'] & zscore['Child_Factual_var2'])
zscore['within_reporter']=zscore['within_reporter']+(zscore['Family_parent-report_var1'] & zscore['Family_parent-report_var2'])
zscore['within_reporter']=zscore['within_reporter']+(zscore['Child_Self-report_var1'] & zscore['Child_Self-report_var2'])
zscore['within_reporter']=zscore['within_reporter']+(zscore['Child_parent-report_var1'] & zscore['Child_parent-report_var2'])
zscore['within_reporter']=zscore['within_reporter']+(zscore['Parent_parent-report_var1'] & zscore['Parent_parent-report_var2'])

zscore['between_reporter']=abs(zscore['within_reporter']-1)


# In[22]:


#drop transitional variables from zscore dataframe
for x in domains:
    zscore=zscore.drop((str(x)+'_var1'),axis=1)
    zscore=zscore.drop((str(x)+'_var2'),axis=1)
for x in reporters:
    zscore=zscore.drop((str(x)+'_var1'),axis=1)
    zscore=zscore.drop((str(x)+'_var2'),axis=1)
#zscore=zscore.drop(['index','redun','var1','var2'],axis=1)
zscore=zscore.drop(['index','redun'],axis=1)


# In[23]:


#print zscore file to csv for local fdr
zscore.to_csv("owens_espaper_zscore&categories.csv")


# In[24]:


#Calculate Within Reporter Correaltions
i=0
cx_crr_ls=[data_child_parentreport,data_child_selfreport,data_child_task,data_family_parentreport,
           data_parent_parentreport]

names=["Parent Report on Child", "Child Self Report","Child Task",
       "Parent Report on Other Family Members x Parent Self Report","Parent Self Report"]
      
cx_crrs=[]
for v,n in zip(cx_crr_ls,names):
    datacr=v
    datacr=datacr.sort_index(axis=1)
    crr2=datacr.corr(method='pearson')
    crr2=abs(crr2)
    sn.heatmap(crr2,cmap="Reds")
    plt.title("Correlation Among "+str(n))
    plt.xlabel('Variable Names')
    plt.ylabel('Variable Names')
    plt.show()
    crr5 = crr2.stack().reset_index()
    #set column names
    crr5['2scor']=crrb['cor']
    crr5.columns = ['var1','var2','cor','2scor']
    crr5=crr5[crr5['cor']<1]
    crr5.drop_duplicates(subset ="cor", 
                         keep = "first", inplace = True) 
    crr5=crr5.sort_values(by='cor',ascending=False).reset_index()
    if i == 0:
        crr6=crr5
    else:
        #crr6.append(crr5)
        crr6=pd.concat([crr6,crr5])
    plt.hist(crr5['cor'],bins=50)
    plt.title(str(n)+" - absolute value")
    plt.xlabel('Bivariate Correlation Value')
    plt.ylabel('Number of Pairs with a given Correlation')
    plt.show()
    cx_crrs.append(crr5)
    #Quantiles
    i+=1
    print("Quantiles for"+str(n))
    print(crr5['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95]))
    agg_quart[str(n)]=crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
    print(crr5.iloc[:,1:4].head(15))
crr6.to_csv("stacked_cormat_within_reporter.csv")
plt.hist(crr6['cor'],bins=100)
plt.title("Within Reporter Correlations")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
#Quantiles
i+=1
print("Quantiles for"+str(n))
agg_quart['within-reporter']=crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
print(crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95]))
print(crr6.iloc[:,1:4].head(15))


# In[25]:


#Calculate Between Reporter Anlayses
i=0
cx_crr_ls=[cpr_csr,cpr_ct,cpr_fpr,cpr_ppr,csr_ct,csr_fpr,csr_ppr,ct_fpr,ct_ppr,fpr_ppr]

names=["Parent Report on Child x Child Self Report","Parent Report on Child x Child Task",
       "Parent Report on Child x Parent Report on Other Family Members", "Parent Report on Child vs Parent Self Report",
      "Child Self Report x Child Task","Child Self Report x Parent Report on Other Family Members",
      "Child Self Report x Parent Self Report","Child Task x Parent Report on Other Family Members",
      "Child Task x Parent Self Report","Parent Report on Other Family Members x Parent Self Report"]
cx_crrs=[]
for v,n in zip(cx_crr_ls,names):
    crr2=abs(v)
    sn.heatmap(crr2,cmap="Reds")
    plt.title("Correlation Among "+str(n))
    plt.xlabel('Variable Names')
    plt.ylabel('Variable Names')
    plt.show()
    crr5 = crr2.stack().reset_index()
    #set column names
    crr5['2scor']=crrb['cor']
    crr5.columns = ['var1','var2','cor','2scor']
    crr5=crr5[crr5['cor']<1]
    crr5.drop_duplicates(subset ="cor", 
                         keep = "first", inplace = True) 
    crr5=crr5.sort_values(by='cor',ascending=False).reset_index()
    if i == 0:
        crr6=crr5
    else:
        #crr6.append(crr5)
        crr6=pd.concat([crr6,crr5])
    plt.hist(crr5['cor'],bins=50)
    plt.title(str(n)+" - absolute value")
    plt.xlabel('Bivariate Correlation Value')
    plt.ylabel('Number of Pairs with a given Correlation')
    plt.show()
    cx_crrs.append(crr5)
    #Quantiles
    i+=1
    print("Quantiles for"+str(n))
    print(crr5['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95]))
    agg_quart[str(n)]=crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
    print(crr5.iloc[:,1:4].head(15))
crr6.to_csv("stacked_cormat_cross_reporter.csv")
plt.hist(crr6['cor'],bins=50)
plt.title("Between Reporter Correlations")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
cx_crrs.append(crr6)
#Quantiles
i+=1
print("Quantiles for"+str(n))
agg_quart['cross-reporter']=crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
print(crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95]))


# In[26]:


#Calculate Between Domain Anlayses
cx_crr_ls=[mh_sd,mh_bio,mh_cog,mh_per,mh_fam,sd_bio,sd_cog,sd_per,sd_fam,bio_cog,bio_per,bio_bam,
           cog_per,cog_fam,per_fam]
names=["Mental Health x Sociodemographic","Mental Health x Biological","Mental Health x Cognitive",
      "Mental Health x Personality","Mental Health x Family/Social","Sociodemographic x Biological",
      "Sociodemographic x Cognitive","Sociodemographic x Personality","Sociodemographic x Family/Social",
      "Biological x Cognitive","Biological x Personality","Biological x Family/Social","Cognitive x Personality",
      "Cognitive x Family/Social","Personality x Family/Social"]
cx_crrs=[]
for v,n in zip(cx_crr_ls,names):
    crr2=abs(v)
    sn.heatmap(crr2,cmap="Reds")
    plt.title("Correlation Among "+str(n))
    plt.xlabel('Variable Names')
    plt.ylabel('Variable Names')
    plt.show()
    crr5 = crr2.stack().reset_index()
    #set column names
    crr5['2scor']=crrb['cor']
    crr5.columns = ['var1','var2','cor','2scor']
    crr5=crr5[crr5['cor']<1]
    crr5.drop_duplicates(subset ="cor", 
                         keep = "first", inplace = True) 
    crr5=crr5.sort_values(by='cor',ascending=False).reset_index()
    if i == 0:
        crr6=crr5
    else:
        #crr6.append(crr5)
        crr6=pd.concat([crr6,crr5])
    plt.hist(crr5['cor'],bins=50)
    plt.title(str(n)+" - absolute value")
    plt.xlabel('Bivariate Correlation Value')
    plt.ylabel('Number of Pairs with a given Correlation')
    plt.show()
    cx_crrs.append(crr5)
    #Quantiles
    i+=1
    print("Quantiles for"+str(n))
    print(crr5['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95]))
    agg_quart[str(n)]=crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
    print(crr5.iloc[:,1:4].head(15))
crr6.to_csv("stacked_cormat_cross_domain.csv")
plt.hist(crr6['cor'],bins=100)
plt.title("Between Domain Correlations")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
cx_crrs.append(crr6)
#Quantiles
i+=1
print("Quantiles for all between domain")
print(crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95]))
agg_quart['cross-domain']=crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
print(crr6.iloc[:,1:4].head(15))


# In[27]:


#Calcualte Within Domain Analyses
i=0
cx_crr_ls=[data_mentalhealth,data_sociodemographic,data_biological,
           data_cognitive,data_personality,data_family]

names=["Meantal Health Measures","Sociodemographic Measures","Biological Measures",
       "Cognitive Tasks",'Personality Measures', "Family Measures"]
cx_crrs=[]
for v,n in zip(cx_crr_ls,names):
    datacr=v
    datacr=datacr.sort_index(axis=1)
    crr2=datacr.corr(method='pearson')
    crr2=abs(crr2)
    sn.heatmap(crr2,cmap="Reds")
    plt.title("Correlation Among "+str(n))
    plt.xlabel('Variable Names')
    plt.ylabel('Variable Names')
    plt.show()
    crr5 = crr2.stack().reset_index()
    #set column names
    crr5['2scor']=crrb['cor']
    crr5.columns = ['var1','var2','cor','2scor']
    crr5=crr5[crr5['cor']<1]
    crr5.drop_duplicates(subset ="cor", 
                         keep = "first", inplace = True) 
    crr5=crr5.sort_values(by='cor',ascending=False).reset_index()
    if i == 0:
        crr6=crr5
    else:
        #crr6.append(crr5)
        crr6=pd.concat([crr6,crr5])
    plt.hist(crr5['cor'],bins=50)
    plt.title(str(n)+" - absolute value")
    plt.xlabel('Bivariate Correlation Value')
    plt.ylabel('Number of Pairs with a given Correlation')
    plt.show()
    cx_crrs.append(crr5)
    #Quantiles
    i+=1
    print("Quantiles for "+str(n))
    print(crr5['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95]))
    agg_quart[str(n)]=crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
    print(crr5.iloc[:,1:4].head(15))
crr6.to_csv("stacked_cormat_within_domain.csv")
plt.hist(crr6['cor'],bins=100)
plt.title("Within Domain Correlations")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
cx_crrs.append(crr5)
#Quantiles
i+=1
print("Quantiles for all within domain")
print(crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95]))
agg_quart['within-domain']=crr6['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
print(crr6.iloc[:,1:4].head(15))


# In[28]:


#drop covariates from variable list for partial correlation analyses
covs=['age','sex','high.educ.bl','household.income.bl','married.bl','hisp','race.6level']
#data_covvars=datacr[covs]
#datacovs=datacr.drop(covs, axis=1)
varr=data_fin.columns
varr2=varr.drop(covs)

#add scanner to dataframe for covariate analysis
dummy_df = pd.get_dummies(scanner)
data_fin_wscanner = pd.concat([data_fin, dummy_df], axis = 1)
scanner_dummyvars = list(dummy_df.columns)


# In[29]:


#Run partial correlations (using NDA covariates)
covs_wscanner=['age','sex','high.educ.bl','household.income.bl','married.bl','hisp','race.6level'] + scanner_dummyvars
varr2=varr.drop('demo_prtnr_empl_p')
varr2=varr2.drop('demo_prnt_empl_time_p')
varr2=varr2.drop('devhx_milestones')
varr2=varr2.drop('devhx_23b_age_wet_bed_p')
#comat=pingouin.pairwise_corr(data=data_fin_wscanner,columns=varr2,covar=covs_wscanner,method='pearson',tail='two-sided',nan_policy='pairwise')
#comat.to_csv("partial_corr_matrix_wscanner.csv")


# In[30]:


#calculate p-values for thresholding analyses without covariates
#comat=pingouin.pairwise_corr(data=data_fin,columns=varr,covar=None,method='pearson',tail='two-sided',nan_policy='pairwise')
#comat.to_csv("partial_corr_matrix_wscanner.csv")


# In[31]:


#read in partial corrs so you don't have to wait for them to run every time you run analyses
core=pd.read_csv('corr_matrix_wpvals2.csv')
pcore=pd.read_csv('partial_corr_matrix_wscanner.csv')


# In[32]:


#redo main anlaysis from above to confirm new cor tool works the same
rsa=core.iloc[:,[1,2,5,6,11]]
#set column names
rsa.columns = ['var1','var2','n','cor','pvals']
rsa['cor']=abs(rsa['cor'])
rsa['sorted_row'] = [sorted([a,b]) for a,b in zip(rsa.var1, rsa.var2)]
rsa['sorted_row'] = rsa['sorted_row'].astype(str)
rsa.drop_duplicates(subset=['sorted_row'], inplace=True)  
rsa=rsa.sort_values(by='cor',ascending=False).reset_index()
rsa.to_csv("stacked_cormat_abs.csv")
plt.hist(rsa['cor'],bins=100)
plt.title("All Variables - absolute value")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
#Quantiles
rsa.quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95],axis=0)


# In[33]:


#set up fdr - https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
import statsmodels.api
fdr=statsmodels.stats.multitest.multipletests(rsa['pvals'],alpha=0.05,method='fdr_bh',returnsorted=False)
rsa['fdr-rej']=DataFrame(fdr[0])
rsa['fdr-p']=DataFrame(fdr[1])
rsa_fdr=rsa
#rsa_fdr.to_csv('fdr_fulldata.csv')
#Drop non-significant correlations from fdr
rsa_fdr_t=rsa_fdr[rsa_fdr['fdr-rej']==True]

bon=statsmodels.stats.multitest.multipletests(rsa['pvals'],alpha=0.05,method='bonferroni',returnsorted=False)
rsa['bon-rej']=DataFrame(bon[0])
rsa['bon-p']=DataFrame(bon[1])
rsa_bon=rsa
#Drop non-significant correlations from fdr
rsa_bon_t=rsa_bon[rsa_bon['bon-rej']==True]
plt.hist(rsa_fdr_t['cor'],bins=100)
plt.title("All Variables - absolute value")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
plt.hist(rsa_bon_t['cor'],bins=100)
plt.title("All Variables - absolute value")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
#Quantiles
agg_quart['fdr']=rsa_fdr_t['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
agg_quart['bon']=rsa_bon_t['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
print(rsa_fdr_t.quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95],axis=0))
print(rsa_bon_t.quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95],axis=0))


# In[34]:


#Drop non-significant correlations
rsa=core.iloc[:,[1,2,6,11]]
#set column names
rsa.columns = ['var1','var2','cor','pvals']
rsa['cor']=abs(rsa['cor'])
rsa['sorted_row'] = [sorted([a,b]) for a,b in zip(rsa.var1, rsa.var2)]
rsa['sorted_row'] = rsa['sorted_row'].astype(str)
rsa.drop_duplicates(subset=['sorted_row'], inplace=True) 
rsa=rsa.sort_values(by='cor',ascending=False).reset_index()
rsat=rsa[rsa['pvals']<.05]
rsa.to_csv("stacked_cormat_abs.csv")
plt.hist(rsat['cor'],bins=100)
plt.title("All Variables")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
#Quantiles
agg_quart['.05']=rsat['cor'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
rsa.quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95],axis=0)


# In[35]:


#show sig and non-sig correlations
rsa=core.iloc[:,[1,2,6,11]]
#set column names
rsa.columns = ['var1','var2','cor','pvals']
rsa['cor']=abs(rsa['cor'])
rsa['sorted_row'] = [sorted([a,b]) for a,b in zip(rsa.var1, rsa.var2)]
rsa['sorted_row'] = rsa['sorted_row'].astype(str)
rsa.drop_duplicates(subset=['sorted_row'], inplace=True) 
rsa=rsa.sort_values(by='cor',ascending=False).reset_index()
rsat=rsa[rsa['pvals']<.05]
#rsa.to_csv("stacked_cormat_abs.csv")
plt.hist(rsa['cor'],bins=100)
plt.hist(rsat['cor'],bins=100,color='r')
plt.hist(rsa_fdr_t['cor'],bins=100,color='g')
plt.hist(rsa_bon_t['cor'],bins=100,color='y')
plt.title("All Variables")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
#Quantiles
rsa_bon_t.quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95],axis=0)
rsa_bon_t['cor'].min()


# In[36]:


#Make figure showing all thresholding analyses
plt.hist(rsa['cor'],bins=100)
plt.hist(rsat['cor'],bins=100,color='r')
plt.hist(rsa_fdr_t['cor'],bins=100,color='g')
plt.hist(rsa_bon_t['cor'],bins=100,color='y')
plt.legend(['Unthresholded', 'p<.05', 'FDR', 'Bonferroni'])
plt.title("All Variables")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
#Quantiles
rsa_bon_t.quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95],axis=0)
rsa_bon_t['cor'].min()


# In[37]:


#test results for only continuous data types
dtypes=pd.read_csv('es_dtypes3.csv')
contin=dtypes[dtypes['types']=='c']
contin=contin['names']
data_contin=data[contin]
data_contin=data_contin.dropna(axis=1,how="all")
data_contin=data_contin.sort_index(axis=1)
crr=data_contin.corr(method='pearson')
crr=abs(crr)
crr4 = crr.stack().reset_index()
#set column names
crr4['2scor']=crrb['cor']
crr4.columns = ['var1','var2','cor','2scor']
crr4=crr4[crr4['cor']<1]
crr4['sorted_row'] = [sorted([a,b]) for a,b in zip(crr4.var1, crr4.var2)]
crr4['sorted_row'] = crr4['sorted_row'].astype(str)
crr4.drop_duplicates(subset=['sorted_row'], inplace=True)
crr4=crr4.sort_values(by='cor',ascending=False).reset_index()
crr4.to_csv("stacked_cormat_abs_pd.csv")
plt.hist(crr4['cor'],bins=100)
plt.title("All Variables - absolute value")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.show()
crr_fin=crr4
#Quantiles
agg_quart['continuous']=crr4['cor'].quantile(q=[.1,.1665,.2,.25,.3,.4,.5,.6,.7,.75,.8,.8335,.9,.95])
crr4.quantile(q=[.1,.1665,.2,.25,.3,.4,.5,.6,.7,.75,.8,.8335,.9,.95],axis=0)


# In[38]:


#full and partial correlations without thresholding
prsa=pcore.iloc[:,[1,2,7,12]]
#set column names
prsa.columns = ['var1','var2','cor','pvals']
prsa['cor']=abs(prsa['cor'])
prsa['sorted_row'] = [sorted([a,b]) for a,b in zip(prsa.var1, prsa.var2)]
prsa['sorted_row'] = prsa['sorted_row'].astype(str)
prsa.drop_duplicates(subset=['sorted_row'], inplace=True)
prsa=prsa.sort_values(by='cor',ascending=False).reset_index()
prsat=prsa[prsa['pvals']<.05]
#rsa.to_csv("stacked_cormat_abs.csv")
plt.hist(rsa['cor'],bins=100)
plt.hist(prsa['cor'],bins=100,color='r')
plt.legend(['Full Correlation', 'Partial Correlation'])
plt.title("All Variables")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.savefig('Figure_2.tiff', dpi=300)
plt.show()
#Quantiles
prsa.quantile(q=[.1,.1665,.2,.25,.3,.4,.5,.6,.7,.75,.8,.8335,.9,.95],axis=0)


# In[42]:


#full and partial correlations + MLE models without thresholding
lme_core=pd.read_csv('ES_MLM_stats.csv')
lme_core=lme_core.iloc[:,[1,2,6,7,8]]
#set column names
lme_core.columns = ['var1','var2','pvals','R2','N']
lme_core[lme_core['R2'] < 0] = 0
lme_core['r'] = np.sqrt(lme_core['R2'])
lme_core['sorted_row'] = [sorted([a,b]) for a,b in zip(lme_core.var1, lme_core.var2)]
lme_core['sorted_row'] = lme_core['sorted_row'].astype(str)
lme_core.drop_duplicates(subset=['sorted_row'], inplace=True)
lme_core = lme_core.sort_values(by='r',ascending=False).reset_index()
#lme_core_t=lme_core[lme_core['pvals']<.05]
plt.hist(rsa['cor'],bins=100)
plt.hist(prsa['cor'],bins=100,color='r')
plt.hist(lme_core['r'],bins=100,color='y')
plt.legend(['Full Correlation', 'Partial Corelation', 'Mixed Effect (approx) Correlation'])
plt.title("All Variables")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.savefig('Figure_xx_RR.tiff', dpi=300)
plt.show()
#Quantiles
lme_core['r'].quantile(q=[.1,.1665,.2,.25,.3,.4,.5,.6,.7,.75,.8,.8335,.9,.95])


# In[44]:


#final integrative "real-world" analyses

#import mixed effects results back from R
lme_core=pd.read_csv('ES_MLM_stats.csv')
lme_core=lme_core.iloc[:,[1,2,6,7,8]]

#set column names
lme_core.columns = ['var1','var2','pvals','R2','N']
lme_core[lme_core['R2'] < 0] = 0
lme_core['r'] = np.sqrt(lme_core['R2'])

#remove within scale correlations
lme_core['redun'] = lme_core['var1'].astype(str).str[0:3]==lme_core['var2'].astype(str).str[0:3]
cr=cr[cr['redun']==False]
lme_core['sorted_row'] = [sorted([a,b]) for a,b in zip(lme_core.var1, lme_core.var2)]
lme_core['sorted_row'] = lme_core['sorted_row'].astype(str)
lme_core.drop_duplicates(subset=['sorted_row'], inplace=True)
lme_core = lme_core.sort_values(by='r',ascending=False).reset_index()

#set up thrsholding
lme_core_t=lme_core[lme_core['pvals']<.05]
fdr=statsmodels.stats.multitest.multipletests(lme_core['pvals'],alpha=0.05,method='fdr_bh',returnsorted=False)
lme_core['fdr-rej']=DataFrame(fdr[0])
lme_core['fdr-p']=DataFrame(fdr[1])
lme_core_fdr=lme_core
lme_core_fdr_t=lme_core_fdr[lme_core_fdr['fdr-rej']==True]

bon=statsmodels.stats.multitest.multipletests(lme_core['pvals'],alpha=0.05,method='bonferroni',returnsorted=False)
lme_core['bon-rej']=DataFrame(bon[0])
lme_core['bon-p']=DataFrame(bon[1])
lme_core_bon=lme_core
lme_core_bon_t=lme_core_bon[lme_core_bon['bon-rej']==True]

plt.hist(lme_core_fdr_t['r'],bins=100)
plt.title("Real World")
plt.xlabel('Bivariate Correlation Value')
plt.ylabel('Number of Pairs with a given Correlation')
plt.savefig('Figure_4_RR.tiff', dpi=300)
plt.show()
#Quantiles
agg_quart['real world']=lme_core_fdr_t['r'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95])
lme_core_fdr_t['r'].quantile(q=[.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99])


# In[45]:


agg_quart.to_csv('agg.csv')


# ## Alpha Analysis

# In[46]:


#redo main anlaysis from above to confirm new cor tool works the same
alf=rsa

#alf.rename(alf['var1'])
alf=pd.merge(alf,alphas,how='inner',on=['var1'])
alf=alf.drop(['var2_y'],axis=1)
alf.columns = ['index','var1','var2','cor','pvals','sorted_row','alpha1']
alf=alf.drop('index',axis=1)
alf=pd.merge(alf,alphas,how='inner',on=['var2'])
alf=alf.drop(['var1_y'],axis=1)
alf.columns = ['var1','var2','cor','pvals','sorted_row','alpha1','alpha2']
alf['alpha_prod']=alf['alpha1']*alf['alpha2']
apc_sig=pingouin.pairwise_corr(data=alf,columns=['alpha_prod', 'cor'],covar=None,method='pearson',tail='two-sided',nan_policy='pairwise')
apc_sig


# In[47]:


plt.hist(alphas[0])
plt.title("Distribution of Cronbach's Alphas")
plt.xlabel('Alpha')
plt.ylabel('Count')
plt.savefig('Figure_RR.tiff', dpi=300)


# In[48]:


print(alphas[0].quantile(q=[.1,.25,.5,.75,.9]))
print(alphas[0].min())
print(alphas[0].max())


# In[49]:


#count number of continuous vars
len(alf['var1'].unique())


# ### check missingness

# In[50]:


missingness=data_fin.isna().sum()
mssrt=missingness.sort_values(ascending=False)
mssrt[mssrt>1]
#mssrt.to_csv('/home/max/Dropbox/ABCD/DRD/missingvals.csv')


# In[51]:


#count number of tests in multiple comparison correction
len(rsa_bon)
len(rsa_fdr)


# In[ ]:




