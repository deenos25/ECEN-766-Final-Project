import zipfile
import pandas as pd
import numpy as np
from secondary_struct_info import predict_secondary_structure

# Open data file to obtain gene, variant, clinical significance, LLR file id, and ESM1b score
ClinVar_path = 'G:/Shared drives/766 Project/data/ClinVar_ALL_isoform_missense_predictions.csv'
with open(ClinVar_path, 'r') as file:
    ClinVar_df = pd.read_csv(file)

class RetrieveData:
    '''Collect data from files to train network'''
    def __init__(self, id, ClinVar_df):
        self.id = id
        self.ClinVar_df = ClinVar_df

    def import_LLR_df(self):
        # Open data file to retrieve ESM1b LLR for the gene sequence
        zip_path = r'G:\Shared drives\766 Project\data\ALL_hum_isoforms_ESM1b_LLR.zip'
        LLR_csv = f'content/ALL_hum_isoforms_ESM1b_LLR/{self.id}_LLR.csv'
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open(LLR_csv) as file:
                LLR_df = pd.read_csv(file)
        return LLR_df

    def import_data_for_model(self, window_size=10):
        W = int(window_size / 2)
        LLR_df = self.import_LLR_df()
        LLR_file_id_df = self.ClinVar_df.loc[self.ClinVar_df['LLR_file_id'] == self.id]
        results = []
        for i in LLR_file_id_df['variant']:
            # Locate variant location in gene
            variant_loc = f"{i[:1]} {i[1:-1]}"
            seq_center_at_variant_loc = int(np.where(LLR_df.columns == variant_loc)[0])
            # Retrieve neighboring residue sequence and LLR information
            if (seq_center_at_variant_loc - W) < 1:
                start_index = 1
                end_index = W*2 + 2
            elif (seq_center_at_variant_loc + W + 1) > len(LLR_df.columns):
                end_index = len(LLR_df.columns)
                start_index = len(LLR_df.columns) - (W*2 + 1)
            else:
                start_index = seq_center_at_variant_loc - W
                end_index = seq_center_at_variant_loc + W + 1
            seq_lst = LLR_df.columns[start_index:end_index]
            seq_LLR_arr = LLR_df.iloc[:, start_index:end_index].to_numpy()
            variant_loc_LLR_arr = LLR_df.iloc[:, seq_center_at_variant_loc].to_numpy()
            # Convert sequence list to string and predict secondary structure fractions
            seq_letters = ''.join([seq.split()[0] for seq in seq_lst])
            seq_arr = np.array([seq.split()[0] for seq in seq_lst]).T
            pred_2_struct_arr_wt = np.array(list(predict_secondary_structure(seq_letters).values()))
            seq_lst_v = LLR_df.columns[start_index:end_index]
            seq_lst_v = [seq.split()[0] for seq in seq_lst_v]
            seq_lst_v[W] = i[-1]
            seq_letters_v = ''.join(seq_lst_v)
            pred_2_struct_arr_v = np.array(list(predict_secondary_structure(seq_letters_v).values()))
            # Get amino acid at variant position and also variant amino acid
            real_acid, variant_acid, position = i[0], i[-1], int(i[1:-1])
            # Get clinical significance label
            label = list(LLR_file_id_df['ClinicalSignificance'].loc[LLR_file_id_df['variant'] == i])[0]
            # Get ESM1b score aka LLR for variant
            esm1b_score = float(LLR_file_id_df.loc[LLR_file_id_df['variant'] == i, 'ESM1b_score'].iloc[0])
            if len(seq_lst) != (W*2) + 1:
                print('Error', len(seq_lst))
                break
            results.append((real_acid, variant_acid, position, esm1b_score, seq_arr, pred_2_struct_arr_wt, pred_2_struct_arr_v, variant_loc_LLR_arr, seq_LLR_arr, label))
        return results

# Usage
len_of_data = 0
labels_to_keep = ['Benign', 'Likely benign', 'Likely pathogenic', 'Pathogenic']
ClinVar_df_filtered = ClinVar_df[ClinVar_df['ClinicalSignificance'].isin(labels_to_keep)]
model_data = np.empty(10)
for id in ClinVar_df_filtered['LLR_file_id'].unique():
    print(id)
    try:
        results = RetrieveData(id, ClinVar_df_filtered).import_data_for_model(window_size=50)
        len_of_data += len(results)
        print(len_of_data)
    except:
        continue
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    model_data = np.row_stack((model_data, results))
col_names = ['real_acid', 'variant_acid', 'position', 'esm1b_score', 'seq_arr', 'pred_2nd_struct_arr_wt', 'pred_2nd_struct_arr_v', 'variant_loc_LLR_arr', 'seq_LLR_arr', 'label']
extracted_data_df = pd.DataFrame(model_data[1:], columns=col_names)
extracted_data_df.to_pickle('extracted_data_W51_v2.pkl')