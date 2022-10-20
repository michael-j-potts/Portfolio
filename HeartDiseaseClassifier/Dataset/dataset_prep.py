import numpy as np
import pandas as pd
import sys
import pickle


def data_organizer(input_file):
    file = []
    for line in input_file:
        line = line.replace('\n', "")
        split = line.split(" ")
        file.append(split)
    data = []
    patient = []
    for i in file:
        for j in i:
            if j != 'name':
                patient.append(j)
            else: 
                data.append(patient)
                patient = []
    return data

def percent_total_column_data(patient_list):
    remove_list = []
    null_count = 0
    value_count = 0
    for column in patient_list:
        for i in patient_list[column]:
            if i == str(-9) or i == '-9.':
                null_count += 1
            else:
                value_count += 1
        percent = value_count / (value_count + null_count)
        #print('Value data in column: ', column, percent, '%')
        if percent < 0.9:
            remove_list.append(column)
        null_count = 0
        value_count = 0
    return remove_list

def percent_total_row_data(patient_list):
    remove_list = []
    null_count = 0
    value_count = 0
    for row in np.transpose(patient_list):
        for i in np.transpose(patient_list)[row]:
            if i == str(-9) or i == '-9.':
                null_count += 1
            else:
                value_count += 1
        percent = value_count / (value_count + null_count)
        #print('Value data in column: ', column, percent, '%')
        if percent < 0.9:
            remove_list.append(row)
        null_count = 0
        value_count = 0
    return remove_list

def description(patient_list):
    #Some initial patient description:
    print("Current number of patients: ", len(patient_list), ", Original number of patients: ", len(original_patient_list))
    print("Percent of original patients with viable data: ", np.round(len(patient_list.index)/ len(original_patient_list.index), 3))
    print("Patients with heart disease: ", np.count_nonzero(patient_list.heart_dis), " or ", np.round((np.count_nonzero(patient_list.heart_dis)/
            len(patient_list) * 100), 3), "% of patients")

    #look at the descriptive statistics for our patient list that involve numeric values (not including binary or categorical values)
    # #xxxxx variables represents columns that were dropped when the limit for column data retention rate was increased from 80% to 90%
    description_data = patient_list[['age', 'admit_systolic_bp', 'chole', 'max_hr_achieved', 'rest_hr', 'exer_dur',#'peak_exer_systolic', 'peak_exer_diastolic',
                'resting_systolic', 'resting_diastolic']]

    #Only 3 of 7 columns are viable as many rows include 0's. need to remove ****************************************************
    print(np.round(np.transpose(description_data.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])))
    return description_data

data1 = data_organizer(open('./hungarian.data.txt', 'r'))
data2 = data_organizer(open('./long-beach-va.data.txt', 'r'))
data3 = data_organizer(open('./switzerland.data.txt', 'r'))
data4 = data_organizer(open('./cleveland.data.txt', 'r'))

dataset = data1 + data2 + data3 + data4
original_patient_list = pd.DataFrame(dataset, columns = ['id', 'social_security', 'age', 'sex', 'pain_loc', 'pain_exert', 'rel_rest',
                        'pncaden', 'chest_pain_type', 'admit_systolic_bp', 'hypertension', 'chole', 'smoker', 'cigs_day', 'smoker_years',
                        'fasting_glucose', 'diabetes', 'fam_hist_cor_art_dis', 'rest_ecg', 'exer_ecg_month', "exer_ecg_day", 'exer_ecg_year',
                        'digitalis_ecg', 'prop_ecg', 'nitr_ecg', 'ccb_ecg', 'diur_ecg', 'exer_protocol', 'exer_dur', 'st_dep_time', 'met',
                        'max_hr_achieved', 'rest_hr', 'peak_exer_systolic', 'peak_exer_diastolic', 'resting_systolic', 'resting_diastolic', 'exer_ang', 'xhypo', 'old_peak', 
                        'slope', 'rldv5', 'rldv5e', 'flour_res', 'restckm', 'exerckm', 'rest_eject_frac', 'rest_wall_move', 'exer_eject_frac',
                        'exer_wall_move', 'thal', 'thalsev', 'thalpul', 'earlobe', 'card_cath_month', 'card_cath_day', 'card_cath_year',
                        'heart_dis', 'lmt', 'ladprox', 'laddist', 'diag', 'cxmain', 'ramus', 'om1', 'om2', 'rcaprox', 'rcadist', 'lvx1',
                        'lvx2', 'lvx3', 'lvx4', 'lvf', 'cathef', 'junk'])

# Drop any extra columns that arent necessary and reset patient id to the order that patients appear in the dataset
patient_list = original_patient_list.drop(columns = ['social_security', 'card_cath_month', 'card_cath_year','card_cath_day', 'junk',
                                        'exer_ecg_month', 'exer_ecg_day', 'exer_ecg_year', 'digitalis_ecg', 'prop_ecg', 'nitr_ecg',
                                        'ccb_ecg', 'diur_ecg', 'lvx1', 'lvx2', 'lvx3', 'lvx4', 'lvf'])
patient_list.id = patient_list.index + 1

#Remove columns with less than x% data fill, then remove rows with less than x% remaining column data fill
remove_column_list = percent_total_column_data(patient_list)
patient_list = patient_list.drop(columns = remove_column_list)
remove_row_list = percent_total_row_data(patient_list)
patient_list = np.transpose(np.transpose(patient_list).drop(columns = remove_row_list)).reset_index(drop = True)

#Replace -9 and -9. values with nans, and convert list to numeric in preparation for converting nans to the column mean values
patient_list = patient_list.replace('-9', np.nan)
patient_list = patient_list.replace('-9.', np.nan)
patient_list = patient_list.apply(pd.to_numeric)

#Describe data 
description_data = description(patient_list)

#replace missing numeric only data with the mean of said column. Not included are binary or categorical number data columns.
#ADD all numeric columns!! ****************************************************************
patient_list[['age', 'admit_systolic_bp', 'chole', 'max_hr_achieved', 'rest_hr', 'exer_dur',
            'resting_systolic', 'resting_diastolic']] = description_data[['age', 'admit_systolic_bp', 'chole', 'max_hr_achieved', 'rest_hr', 'exer_dur',
            'resting_systolic', 'resting_diastolic']].fillna(value = np.round(description_data[['age', 'admit_systolic_bp', 'chole', 'max_hr_achieved', 'rest_hr', 'exer_dur',
            'resting_systolic', 'resting_diastolic']].mean(), 0))

#**********missing values in columns hypertension, xhypo, old_peak...
#**********may need to split variables into dummy nodes

patient_list = patient_list.fillna(value = np.round(patient_list.mean(), 1))

#print(patient_list)

patient_list.heart_dis = patient_list.heart_dis.astype(bool).astype(int)

patient_list.to_csv('../Cleaned_Dataset/Cleaned_Data.csv', index = False)
