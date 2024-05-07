# Repository Contents
## Script Files
* secondary_struct_info.py
* extract_data.py
* model_v1.py
* model_v2.py
* model_v3.py
## File Descriptions
* Output from "secondary_struct_info.py" is the predicted secondary structure content of a protein sequence. It returns back the fraction percentage of predicted secondary structures such as alpha helix, beta sheet, and turn/loop.
* "extract_data.py" &  "extract_dataV2.py" extracts variant & sequence information such as amino acids & position as features, clinical significance/label of the variant, and log-likelihood ratio features calculated by the ESM1b pLM
* "model_v1.py" is the first proposed model for a binary classification of 'Benign' and 'Pathogenic' with structural information included from the wild type sequence
* "model_v2.py" is the proposed model for a binary classification of 'Benign' and 'Pathogenic' with structural information included from the difference between the wild type sequence & variant sequence
* "model_v3.py" is the proposed model for a binary classification of 'Benign' and 'Pathogenic' without structural information
## Data
* extracted_data_W#.pkl or extracted_data_W#_v2.pkl (Contains dataframe of features & labels, W# = Window size of extracted amino acid sequence around variant)
* Features in Dataframes: 'real_acid', 'variant_acid', 'esm1b_score', 'seq_arr', 'pred_2nd_struct_arr', ['pred_2nd_struct_arr_wt', 'pred_2nd_struct_arr_v'...(In _v2 files)], 'variant_loc_LLR_arr', 'seq_LLR_arr'
* Labels in Dataframe: 'label' includes benign & pathogenic
* Link to "extracted_data_W#.pkl" files: https://zenodo.org/records/10950471?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE5NDhhNzhkLWZlYjktNDE0Ny04MzM3LTQ1NTk1MWI5MjczMCIsImRhdGEiOnt9LCJyYW5kb20iOiI0NmY5NDBhMWZmMDljODZlZjNmM2QzODNkZGYwZmJiOSJ9.tGvUnkbg4jozzmxGPzX8OvbfM7z5G_7YGWFR6ocFsMq9x2k4hnYRoHy7WeVMxXoWYQtRM5olxTckd1r2ftaqbw
* Variant data is from ClinVar data downloaded from (https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz)

