from openai import AzureOpenAI
import os, sys
import openai
import json
import numpy as np
import pickle
import pandas as pd
from time import sleep
import signal
import tiktoken
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm

with open('azure_credentials.json', 'r') as file:
    azure_data = json.load(file)
    api_key = azure_data['API_KEY']
    api_version = azure_data['API_VERSION']
    azure_endpoint = azure_data['AZURE_ENDPOINT']
    azure_deployment_name = azure_data['AZURE_DEPLOYMENT_NAME']

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint = azure_endpoint
    )

deployment_name=azure_deployment_name

UNIQUE_ID_COLUMN_NAME = "ROW_ID"
UNIQUE_TEXT_COLUMN_NAME = "TEXT"
UNIQUE_LABEL_COLUMN_NAMES = ['sdoh_community_present','sdoh_economics','behavior_tobacco']

#Paths to MIMIC_CSVs
MIMIC_ADMISSION_CSV = "data/ADMISSIONS.csv" #Fill in path/to/file with the path to your MIMIC-III folder
MIMIC_NOTEEVENTS_CSV = "data/NOTEEVENTS.csv" #Fill in path/to/file with the path to your MIMIC-III folder
MIMIC_SBDH = "data/MIMIC-SBDH.csv" #Fill in path/to/file with the path to your MIMIC-SBDH folder
MIMIC_D_ICD_DIAGNOSES = "data/D_ICD_DIAGNOSES.csv"
MIMIC_DIAGNOSES_ICD = "data/DIAGNOSES_ICD.csv"

admission_df = pd.read_csv(MIMIC_ADMISSION_CSV)
d_icd_diganoses = pd.read_csv(MIMIC_D_ICD_DIAGNOSES)
diganoses_icd = pd.read_csv(MIMIC_DIAGNOSES_ICD)
notes_df = pd.read_csv(MIMIC_NOTEEVENTS_CSV)


newborn_list = admission_df[admission_df["ADMISSION_TYPE"] == "NEWBORN"].SUBJECT_ID.to_list()

cancer_icd9_codes = d_icd_diganoses[d_icd_diganoses['LONG_TITLE'].str.upper().str.contains('NEOPLASM')].ICD9_CODE.tolist()
cancer_df = diganoses_icd[diganoses_icd['ICD9_CODE'].isin(cancer_icd9_codes)]
cancer_list = cancer_df['SUBJECT_ID'].tolist()

discharge_df = notes_df[notes_df['CATEGORY'] == 'Discharge summary']
cancer = discharge_df[discharge_df['SUBJECT_ID'].isin(cancer_list)]
non_neonatal = cancer[~cancer['SUBJECT_ID'].isin(newborn_list)]

sbdh_data = pd.read_csv(open(MIMIC_SBDH, 'r+', encoding='UTF-8'),encoding='UTF-8', on_bad_lines='warn')
sbdh_data = sbdh_data.rename(columns={'row_id':UNIQUE_ID_COLUMN_NAME})
annotated_list = sbdh_data[UNIQUE_ID_COLUMN_NAME].tolist()

annotated_notes = discharge_df[discharge_df[UNIQUE_ID_COLUMN_NAME].isin(annotated_list)]
annotated_subjects = discharge_df[discharge_df[UNIQUE_ID_COLUMN_NAME].isin(annotated_list)].SUBJECT_ID.to_list()

no_soc_his = []
for index, row in non_neonatal.iterrows():
    if 'social history:' not in row[UNIQUE_TEXT_COLUMN_NAME].lower():
        no_soc_his.append(row[UNIQUE_ID_COLUMN_NAME])
        
final_sdoh_list = non_neonatal[~non_neonatal[UNIQUE_ID_COLUMN_NAME].isin(no_soc_his)]

unnanotated_notes = final_sdoh_list[~final_sdoh_list[UNIQUE_ID_COLUMN_NAME].isin(annotated_list)]

def retrieve_social_history(df):
    replace_texts = []
    for row_id in df[UNIQUE_ID_COLUMN_NAME]:
        patient = df[df[UNIQUE_ID_COLUMN_NAME] == row_id][UNIQUE_TEXT_COLUMN_NAME].iloc[0]
        social_history_start = patient.lower().find('social history:')
        pos_ends = []
        pos_ends.append(patient.lower().find('family history:'))
        pos_ends.append(patient.lower().find('physical exam'))
        pos_ends.append(patient.lower().find('medications:'))
        pos_ends.append(patient.lower().find('hospital course:'))
        pos_ends.append(patient.lower().find('review of systems:'))
        pos_ends = [x for x in pos_ends if x > social_history_start]
        pos_ends.append(social_history_start+500)
        social_history_end = min(pos_ends)
        replace_texts.append((row_id,patient[social_history_start:social_history_end]))
    texts = pd.DataFrame(replace_texts,columns =[UNIQUE_ID_COLUMN_NAME,UNIQUE_TEXT_COLUMN_NAME])
    
    return texts

annotated_sh = retrieve_social_history(annotated_notes)

annotated_sh = pd.merge(annotated_sh,sbdh_data[[UNIQUE_ID_COLUMN_NAME] + UNIQUE_LABEL_COLUMN_NAMES],on=UNIQUE_ID_COLUMN_NAME, how='left')

unannotated_sh = retrieve_social_history(unnanotated_notes)

df = newborn_list = notes_df = discharge_df = non_neonatal = annotated_list = annotated_subjects = no_soc_his = final_sdoh_list = unnanotated = sbdh_data = None

economics_binary = [1 if x == 2 else 0 for x in annotated_sh.sdoh_economics.to_list()]
tobacco_binary = [1 if x == 1 or x == 2 else 0 for x in annotated_sh.behavior_tobacco.to_list()]
annotated_sh = annotated_sh.drop(columns=['sdoh_economics','behavior_tobacco'])
annotated_sh['sdoh_economics'] = economics_binary
annotated_sh['behavior_tobacco'] = tobacco_binary

# Choose a MIMIC task. Only one must be true, two must be false. 
# Default: community
community = True
economics = False
tobacco = False

assert community + economics + tobacco == 1, "One and only one must be True, the other two must be False"

if community:
    task = 'community'
    label_column = "sdoh_community_present"
elif economics:
    task = 'economics'
    label_column = "sdoh_economics"
else:
    task = 'tobacco'
    label_column = "behavior_tobacco"
    
task_prompts = pickle.load(open('MIMIC_TASK_PROMPTS.pkl','rb'))

base_system_message = task_prompts[task]['instructions']
system_message = f"<|im_start|>INSTRUCTIONS:\n{base_system_message.strip()}\n<|im_end|>"
query_message = task_prompts[task]['query']

easy_example_pos = annotated_sh[annotated_sh[UNIQUE_ID_COLUMN_NAME] == task_prompts[task]['examples']['easy_example_pos']].iloc[0].TEXT.replace('\n', ' ').strip()
easy_answer_pos = task_prompts[task]['examples']['easy_answer_pos']
easy_answer_pos_explained = task_prompts[task]['examples']['easy_answer_pos_explained']
easy_example_neg = annotated_sh[annotated_sh[UNIQUE_ID_COLUMN_NAME] == task_prompts[task]['examples']['easy_example_neg']].iloc[0].TEXT.replace('\n', ' ').strip()
easy_answer_neg = task_prompts[task]['examples']['easy_answer_neg']
easy_answer_neg_explained = task_prompts[task]['examples']['easy_answer_neg_explained']

hard_example_pos = annotated_sh[annotated_sh[UNIQUE_ID_COLUMN_NAME] == task_prompts[task]['examples']['hard_example_pos']].iloc[0].TEXT.replace('\n', ' ').strip()
hard_answer_pos = task_prompts[task]['examples']['hard_answer_pos']
hard_answer_pos_explained = task_prompts[task]['examples']['hard_answer_pos_explained']
hard_example_neg = annotated_sh[annotated_sh[UNIQUE_ID_COLUMN_NAME] == task_prompts[task]['examples']['hard_example_neg']].iloc[0].TEXT.replace('\n', ' ').strip()
hard_answer_neg = task_prompts[task]['examples']['hard_answer_neg']
hard_answer_neg_explained = task_prompts[task]['examples']['hard_answer_neg_explained']

example_ids = [task_prompts[task]['examples']['easy_example_pos'],task_prompts[task]['examples']['easy_example_neg'],task_prompts[task]['examples']['hard_example_pos'],task_prompts[task]['examples']['hard_example_neg']]

# Defining a function to create the prompt from the instruction system message, the few-shot examples, and the current query
# The function assumes 'examples' is a list of few-shot examples in dictionaries with 'context', 'query' and 'answer' keys
# Example: examples = [{"context": "Lives with wife, no tobacco, no alcohol, no drugs",
# "query": "Does the social history present tobacco use?", "answer": "No."}]
# The function assumes 'query' is a dictionary containing the current query GPT is expected to answer with 'context' and 'query' keys.
# Example: query = [{"context": "Lives alone, history of 1 ppd, no alcohol use, no drug use", 
# "query": "Does the social history present tobacco use?"}]
def create_prompt(system_message, examples, query):
    prompt = system_message
    if examples != None:
        for example in examples:
            prompt += f"\n<|im_start|>CONTEXT:\n{example['context']}\n<|im_end|>"
            prompt += f"\n<|im_start|>QUERY:\n{example['query']}\n<|im_end|>"
            prompt += f"\n<|im_start|>ANSWER:\n{example['answer']}\n<|im_end|>"
    prompt += f"\n<|im_start|>CONTEXT:\n{query['context']}\n<|im_end|>"
    prompt += f"\n<|im_start|>QUERY:\n{query['query']}\n<|im_end|>"
    prompt += f"\n<|im_start|>ANSWER:\n"
    return prompt

# This function sends the prompt to the GPT model
def send_message(prompt, model_name, max_response_tokens=500):
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=0.5,
        max_tokens=max_response_tokens,
        frequency_penalty=0,
        presence_penalty=0,
        stop=['<|im_end|>']
    )
    return response.choices[0].text.strip()

# timeout handler
def alarm_handler(signum, frame):
    print("Timeout... Retrying.")
    raise Exception()

# Defining a function to estimate the number of tokens in a prompt
def estimate_tokens(prompt):
    cl100k_base = tiktoken.get_encoding("cl100k_base") 

    enc = tiktoken.Encoding( 
        name="chatgpt",  
        pat_str=cl100k_base._pat_str, 
        mergeable_ranks=cl100k_base._mergeable_ranks, 
        special_tokens={ 
            **cl100k_base._special_tokens, 
            "<|im_start|>": 100264, 
            "<|im_end|>": 100265
        } 
    ) 

    tokens = enc.encode(prompt,  allowed_special={"<|im_start|>", "<|im_end|>"})
    return len(tokens)

def prepare_examples(shots, example_hard, example_explained):
    if shots:
        if example_hard:
            context_messages = [hard_example_pos, hard_example_neg]
            if example_explained:
                answer_messages = [hard_answer_pos_explained, hard_answer_neg_explained]
            else:
                answer_messages = [hard_answer_pos, hard_answer_neg]
        else:
            context_messages = [easy_example_pos, easy_example_neg]
            if example_explained:
                answer_messages = [easy_answer_pos_explained, easy_answer_neg_explained]
            else:
                answer_messages = [easy_answer_pos, easy_answer_neg]  

        examples = [{"context": context_messages[0], "query": query_message, "answer": answer_messages[0]},{"context": context_messages[1], "query": query_message, "answer": answer_messages[1]}]

    else:
        examples = None
    
    return examples

# This block sends the test set one by one and gathers responses from GPT
# I tried to paralelize it, but it kept throwing timeout errors from the GPT side, not sure what I was doing wrong
# you can try, if you're up for it. its linear right now and that doesn't even get close to the rate limit


def gpt_annotation(dataframe, examples):
    responses = []
    tokens = []
    
    num_annotations = len(dataframe)
    for idx, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        # sys.stdout.write("\r")
        # sys.stdout.write("{}/{} examples remaining.".format(idx+1, num_annotations))
        # sys.stdout.flush()
        context_query = row[UNIQUE_TEXT_COLUMN_NAME]

        # signal.signal(signal.SIGALRM, alarm_handler)
        context_query = context_query.strip()
        
        query = {"context": context_query, "query": query_message}
        
        prompt = create_prompt(system_message, examples, query)
        
        for attempt in range(4):
            try:
                max_response_tokens = 10

                # signal.alarm(5)
                response = send_message(prompt, deployment_name, max_response_tokens)
                # signal.alarm(0)

                if 'Yes' in response:
                    responses.append((row[UNIQUE_ID_COLUMN_NAME],1))
                elif 'No' in response:
                    responses.append((row[UNIQUE_ID_COLUMN_NAME],0))
            except Exception as error:
                print("An exception occurred:", error)
                continue
            break
    
    return responses


shots = True
example_hard = [False,True]
example_explained = [False,True]
arr1 = np.array([[False, False, False]])
arr2 = np.array(np.meshgrid(shots, example_hard, example_explained)).T.reshape(-1,3)
all_annotation_types = np.concatenate((arr1,arr2))
all_annotation_names = ['ZeroShot','TwoShot-E','TwoShot-H','TwoShot-E+Ex','TwoShot-H+Ex']

ann_name = 'TwoShot-H'
idx = 2
examples = prepare_examples(all_annotation_types[idx][0], all_annotation_types[idx][1], all_annotation_types[idx][2])

print('Annotating with',ann_name)
responses = gpt_annotation(unannotated_sh, examples)
print('\nAnnotation Complete.')

resp_df = pd.DataFrame(responses, columns=[UNIQUE_ID_COLUMN_NAME, label_column])

file_name = f"{task}-gpt-train-unannotated.pkl"
pickle.dump(resp_df,open(file_name, 'wb'))
print("{} training set saved to {}".format(ann_name, file_name))


ann_name = 'TwoShot-H'
idx = 2
examples = prepare_examples(all_annotation_types[idx][0], all_annotation_types[idx][1], all_annotation_types[idx][2])

print('Annotating with',ann_name)
responses = gpt_annotation(annotated_sh, examples)
print('\nAnnotation Complete.')

resp_df = pd.DataFrame(responses, columns=[UNIQUE_ID_COLUMN_NAME, label_column])

file_name = f"{task}-gpt-train-annotated.pkl"
pickle.dump(resp_df,open(file_name, 'wb'))
print("{} training set saved to {}".format(ann_name, file_name))