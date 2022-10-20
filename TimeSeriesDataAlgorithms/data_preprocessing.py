import sys
import pickle
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

#Need to create targets
ICD10_codes = ['B342', 'J00', 'J069', 'J120', 'J121', 'J122', 'J123', 'J128', 'J129', 'J206' 'J210', 'J211',
                'J218', 'J219', 'J22', 'J80', 'J960', 'J969', 'P221', 'P228', 'P229', 'P28.5', 'P2881', 'R060', 'R068',
                'U049', 'U071', 'U072']
hospitalization_codes = str(list(range(99218, 99240)))
mortality_codes = []

def list_check(hosp_id, non_hosp_id):
    print("patients hospitalized: ", len(pd.unique(hosp_id['patient_id'])))
    print("patients not hospitalized: ", len(pd.unique(non_hosp_id['patient_id'])))
    duplicate = 0
    for i in hosp_id:
        if i in non_hosp_id:
            duplicate += 1
    print("Patients that appear in both hospitalized and non_hospitalized list: ", duplicate)

    
def respiratory_hospitalization(information):
    #Creates a list of patients that have been diagnosed with an approved ICD code within
    #7 days of also being diagnosed with an inpatient cpt code in our lists.
    #First organize our data, assign a new column for cpt codes and icd being present in our hosp list
    information = (information.sort_values(['patient_id', 'date']).reset_index(drop = True))
    information = information[['patient_id', 'icd', 'cpt', 'date']]
    information['cpt'] = information['cpt'].astype(str)
    information['icd'] = information['icd'].astype(str)
    information["cpt_flag"] = [1 if ele in hospitalization_codes else 0 for ele in information['cpt']]
    information["resp_flag"] = [1 if any(ele.startswith(i) for i in ICD10_codes) else 0 for ele in information['icd']]
    information['end_date'] = np.nan

    #Iterate through all the data to add a diagnosis timeframe of 7 days
    #For every positive respiratory code, we want to set an end date for when a positive CPT code will
    #indicate hospitalization related to the respiratory illness. Finally, we forward fill all nan values
    #based off the previous value found by patient
    date_end = []
    for index, row in information.iterrows():
        if row['resp_flag'] == 1:
            row['end_date'] = row['date'] + timedelta(days = 7)
        date_end.append(row)
    date_end_fill = pd.DataFrame(date_end)
    date_end_fill['end_date'] = date_end_fill.groupby('patient_id')['end_date'].transform(lambda x: x.fillna(method = 'ffill'))
    
    #Next, we evaluate all patient visits where end_date isnt null (to avoid looking at visits with no previous 
    #respiratory diagnosis). Then, we check if our cpt flag is in the 7 day diagnosis window, and we create a patient
    #list based upon patients who were hospitalized within 7 days of a positive respiratory ICD code.
    count = 0
    resp_hosp_ID_list = []
    
    for index, row in date_end_fill.iterrows():
        if pd.isnull(row['end_date']):
            continue
        else:    
            if row['cpt_flag'] == 1:
                if row['end_date'] > row['date']:
                    if row['patient_id'] not in resp_hosp_ID_list:
                        resp_hosp_ID_list.append(row['patient_id'])
                        count += 1
                    
    #Now gather all visits prior to the hospitalization date                
    date_end_fill['target'] = [1 if ele in resp_hosp_ID_list else 0 for ele in date_end_fill['patient_id']]
    #We now define our two cohorts, hospitalized within 7 days of ICD10 code and those not 
    non_resp_hosp_patients = date_end_fill[date_end_fill['target'] == 0]

    #iterate through our rows by patient id, set a flag for when we cross a hospitalization code
    #inside of our diagnosis window
    resp_hosp_info = []
    prev_row_id = []
    new_patient_flag = False

    for index, row in date_end_fill.iterrows():
        row_id = row['patient_id']
        if row_id != prev_row_id:
            prev_row_id = row_id
            new_patient_flag = False

        #if our patient is in our target group, write all rows before being diagnosed with ICD10 code
        #else, write the row if the cpt_flag is negative and we havent come accross a positive cpt code
        #yet.
        if row['target'] == 1:
            if pd.isnull(row['end_date']):
                resp_hosp_info.append(row)
            else:
                if row['cpt_flag'] == 1:
                    new_patient_flag = True
                if row['cpt_flag'] == 0:
                    if new_patient_flag == False:
                        resp_hosp_info.append(row)
                else:
                    new_patient_flag = True

    #convert back to dataframe
    resp_hosp_info = pd.DataFrame(resp_hosp_info)
    print("Total patients hospitalized with a positive respiratory ICD code: ", count)

    return date_end_fill['target'], resp_hosp_info[['patient_id', 'icd', 'cpt', 'date']], non_resp_hosp_patients[['patient_id', 'icd', 'cpt', 'date']]


def drop_single(input):
    input_size = input.groupby('patient_id').size().reset_index(name = 'size')
    input = input.merge(input_size, how = "inner", on = 'patient_id')
    output = input[input['size'] > 1]
    return output


def data_organizer(input):
    func = lambda a: ','.join(a)
    patients = input.groupby(['patient_id', 'date'], as_index = False).agg({'icd': func})
    patients = patients.groupby('patient_id', as_index = False)
    patient_info = []
    list_of_codes = {}

    for id, patient in patients:
        visit_info = []
        for visit in patient['icd']:
            new_visit = []
            codes = visit.split(',')
            for code in codes:
                if code in list_of_codes:
                    new_visit.append(int(list_of_codes[code]))
                else:
                    list_of_codes[code] = len(list_of_codes)
                    new_visit.append(int(list_of_codes[code]))

            visit_info.append(new_visit)
        
        patient_info.append(visit_info)
    return patient_info


if __name__=='__main__':
    runtime_start = datetime.now()
    hosp_ICDFile = sys.argv[1]
    non_hosp_ICDFile = sys.argv[2]
    hosp_MedFile = sys.argv[3]
    non_hosp_MedFile = sys.argv[4]
    #MedDescFile = sys.argv[5]

    #prepare table of data, convert string dates to datetime dates, remove patients less than 0
    hosp_info = pd.read_csv(hosp_ICDFile, index_col = [0]).drop(columns = ['claim_id', 'zip', 'icd_4'])
    non_hosp_info = pd.read_csv(non_hosp_ICDFile, index_col = [0]).drop(columns = ['claim_id', 'zip', 'icd_4'])
    patient_list = pd.concat([hosp_info, non_hosp_info])
    patient_list['date'] = pd.to_datetime(patient_list['date'])
    patient_list['age'] = patient_list['age'].astype(int)
    patient_list = patient_list[patient_list['age'] >=0]
    patient_list = patient_list[patient_list['age'] <= 2]
    patient_list['icd'] = patient_list['icd'].astype(str)
    patient_list = patient_list[~patient_list['icd'].str.contains('-1')]
    patient_list['target'] = 0

    #prepare medication info
    hosp_med_info = pd.read_csv(hosp_MedFile, index_col = [0]).drop(columns = ['claim_id'])
    non_hosp_med_info = pd.read_csv(non_hosp_MedFile, index_col = [0]).drop(columns = ['claim_id'])
    patient_med_list = pd.concat([hosp_med_info, non_hosp_med_info])
    patient_med_list['date'] = pd.to_datetime(patient_med_list['date'])
    patient_med_list['product_id'] = patient_med_list['product_id'].astype(str)
    
    #TO DO 
    #need to process medications further


    #list of patients who have been hospitalized within 7 days of a positive ICD10 code
    target, resp_hosp_info, non_resp_hosp_info = respiratory_hospitalization(patient_list)

    total_size = len(pd.unique(patient_list['patient_id']))
    resp_hosp_size = len(pd.unique(resp_hosp_info['patient_id']))
    print("remaining cohort after isolating patients with data before being hospitalized with ICD: ", resp_hosp_size)
    print("percent of patients hospitalized within 7 days of ICD 10 code: ", np.round(resp_hosp_size / total_size, 4))

    #Now drop single visit only patients from both cohorts for training purposes
    resp_hosp_multivisits = drop_single(resp_hosp_info)
    non_resp_hosp_multivisits = drop_single(non_resp_hosp_info)

    list_check(resp_hosp_multivisits, non_resp_hosp_multivisits)

    print("Remaining patients with multiple visits (hospitalized for positive ICD10 code): ", len(pd.unique(resp_hosp_multivisits['patient_id'])))
    print("Remaining patients with multiple visits (not hospitalized for positive ICD10 code): ", len(pd.unique(non_resp_hosp_multivisits['patient_id'])))

    #Recombine, and separate target before sending to data_organizer
    resp_hosp_multivisits['target'] = 1
    non_resp_hosp_multivisits['target'] = 0
    final_patient_list = pd.concat([resp_hosp_multivisits, non_resp_hosp_multivisits])
    targets = final_patient_list.groupby('patient_id')['target'].first()

    target = []
    for i in targets:
        target.append(i)

    icd_vocab_size = final_patient_list['icd'].nunique()
    print("icd vocab size: ", icd_vocab_size)

    #Gather patient visit information and create a list of visits per patient
    final_patient_structured = data_organizer(final_patient_list)


    pickle.dump(final_patient_structured, open('patient_visit_icds', 'wb'), -1)
    pickle.dump(target, open('labels', 'wb'), -1)
    #pickle.dump(meds, open('patient_visit_meds', 'wb'), -1)
