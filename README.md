# Social Determinants of Health using AnnotateGPT

This repo will deal with the extraction of the social determinants of health (community, tobacco, economics) from discharge summaries of hospitalized patients using prompt engineering in Large Language Models (LLM) like `GPT-3.5-Turbo`. 

We will use MIMIC-III (https://physionet.org/content/mimiciii/1.4/) and CORAL (https://physionet.org/content/curated-oncology-reports/1.0/) datasets along with the Open AI API (`GPT-3.5-Turbo`). Becasue of the restrictions on accessing MIMIC-III and CORAL data directly by an open-source OpenAI API, we need to access via AZURE OpenAI API. Fill in the Azure credentials in the file `azure_credentials_dummy.json` and rename this file to `azure_credentials.json`

## Methodology

### Download Third-Party Data

1. Download `MIMIC-SBDH.csv` from (https://github.com/hibaahsan/MIMIC-SBDH)
2. Download `ADMISSIONS.csv`, `NOTEEVENTS.csv`, `PATIENTS.csv`, `D_ICD_DIAGNOSES.csv` and `DIAGNOSES_ICD.csv` from (https://physionet.org/content/mimiciii/1.4/)
3. Place these `csv` files in the `data` directory
4. Download the CORAL Oncology repots from (https://physionet.org/content/curated-oncology-reports/1.0/)
    ```Shell
    wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/curated-oncology-reports/1.0/
    ```

### AnnotateGPT for MIMIC-III

1. Run the file `AnnotateGPT_MIMIC.ipynb` for all three SDoH (community, tobacco, economics) to extract social determinants from each discharge summary in MIMIC-III
2. This will generate `XX-XX-gpt-train.pkl` file for each annotation and social determinant of health. Some of these files are stored in the directory `AnnotateGPT_MIMIC-III`
3. To train any XGBoost machine learning model after extracting the SDoH, use the file named `XGBoost Training.ipynb` using the generated files for each annotation and social determinant namely `XX-XX-gpt-train.pkl`
4. Results may vary slightly due to nature of GPT annotation

### AnnotateGPT for CORAL

1. Run the file `AnnotateGPT_CORAL.ipynb` for all three SDoH (community, tobacco, economics) to extract social determinants from breast and pancreatic cancer oncology reports in CORAL
2. This will generate `XX-XX.pkl` file for breast/pancreatic cancer and each social determinant of health in the directory namely `AnnotateGPT_Coral`
3. The experiements of AnnotateGPT are only conduced on CORAL dataset to extract the percentage of missing SDoH from both breast and pancreatic cancer reports. That's why only one annotation type is used to extract the metrics. For more detailed analysis, the notebook needs to be revised to extract SDoH for each type of annotation
4. Results may vary slightly due to nature of GPT annotation