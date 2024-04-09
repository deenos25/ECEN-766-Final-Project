# Repository Contents
## Script Files
* secondary_struct_info.py
* extract_data.py
* model_v1.py
* Expected to add: model_v2.py, model_v3.py, & model_v4.py
## File Descriptions
* Output from "secondary_struct_info.py" is the predicted secondary structure content of a protein sequence. It returns back the fraction percentage of predicted secondary structures such as alpha helix, beta sheet, and turn/loop.
* "extract_data.py" extracts variant & sequence information such as amino acids & position as features, clinical significance/label of the variant, and log-likelihood ratio features calculated by the ESM1b pLM
* "model_v1.py" is the first proposed model for a binary classification of 'Benign' and 'Pathogenic' with structural information included
* Expected to add files will be models trained with or without structural information or will be multiclass(N=4) classification models
## Data
* extracted_data_W10.pkl (Contains dataframe of features & labels, W# = Window size of extracted amino acid sequence around variant)
* Expect more "extracted_data_W#.pkl" with varying window sizes
* Features in Dataframe: 'real_acid', 'variant_acid', 'position', 'esm1b_score', 'seq_arr', 'pred_2nd_struct_arr', 'variant_loc_LLR_arr', 'seq_LLR_arr'
* Labels in Dataframe: 'label' includes benign, likely benign, likely pathogenic, & pathogenic
* Variant data is from ClinVar data downloaded from (https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz)

