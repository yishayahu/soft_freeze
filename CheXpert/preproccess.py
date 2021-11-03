'''Data Preprocessing'''

###################
## Prerequisites ##
###################

import pandas as pd
data_root = '/home/dsi/shaya/chexpert_v1_small/CheXpert-v1.0-small/'

img_type = 'small'

Traindata = pd.read_csv(f'{data_root}train.csv')
Traindata_frt = Traindata[Traindata['Path'].str.contains('frontal')].copy()
Traindata_lat = Traindata[Traindata['Path'].str.contains('lateral')].copy()
Traindata_frt.to_csv(f'{data_root}train_frt.csv', index = False)
Traindata_lat.to_csv(f'{data_root}train_lat.csv', index = False)
print('Train data length(frontal):', len(Traindata_frt))
print('Train data length(lateral):', len(Traindata_lat))
print('Train data length(total):', len(Traindata_frt) + len(Traindata_lat))

Validdata = pd.read_csv(f'{data_root}valid.csv')
Validdata_frt = Validdata[Validdata['Path'].str.contains('frontal')].copy()
Validdata_lat = Validdata[Validdata['Path'].str.contains('lateral')].copy()
Validdata_frt.to_csv(f'{data_root}valid_frt.csv', index = False)
Validdata_lat.to_csv(f'{data_root}valid_lat.csv', index = False)
print('Valid data length(frontal):', len(Validdata_frt))
print('Valid data length(lateral):', len(Validdata_lat))
print('Valid data length(total):', len(Validdata_frt) + len(Validdata_lat))

Testdata = pd.read_csv(f'{data_root}valid.csv')
Testdata_frt = Testdata[Testdata['Path'].str.contains('frontal')].copy() # to avoid SettingWithCopyWarning
Testdata_lat = Testdata[Testdata['Path'].str.contains('lateral')].copy()
Testdata_frt.to_csv(f'{data_root}test_frt.csv', index = False)
Testdata_lat.to_csv(f'{data_root}test_lat.csv', index = False)
print('Test data length(frontal):', len(Testdata_frt))
print('Test data length(lateral):', len(Testdata_lat))
print('Test data length(total):', len(Testdata_frt) + len(Testdata_lat))

# Make testset for 200 studies (use given valid set as test set)
Testdata_frt.loc[:, 'Study'] = Testdata_frt.Path.str.split('/').str[2] + '/' + Testdata_frt.Path.str.split('/').str[3]
Testdata_frt_agg = Testdata_frt.groupby('Study').agg('first').reset_index()
Testdata_frt_agg = Testdata_frt_agg.sort_values('Path')
Testdata_frt_agg = Testdata_frt_agg.drop('Study', axis = 1)
Testdata_frt_agg.to_csv(f'{data_root}test_agg.csv', index = False)
print('Test data length(study):', len(Testdata_frt_agg))