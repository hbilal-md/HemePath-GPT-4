import requests
import json
import pandas as pd
import re
from rich.progress import Progress

#gpt = "gpt-3.5-turbo"
gpt = "gpt-4"

cbcCols = [
    "Abs Lymph",
    "Abs Mono",
    "Abs Neut",
    "Absolute Basophil",
    "Absolute Blasts",
    "Absolute Eosinophil",
    "Absolute Immature Granulocyte",
    "Baso",
    "Blast_y",
    "Eos",
    "HCT",
    "HGB",
    "Immature Granulocyte",
    "Lymph",
    "Mean Corpuscular Hemoglobin (MCH)",
    "Mean Corpuscular Hemoglobin Conc (MCHC)",
    "Mean Corpuscular Volume (MCV)",
    "Mono",
    "Neutrophil",
    "Nucleated RBC",
    "Platelets",
    "RBC",
    "Red Blood Cell Distribution Width (RDW)",
    "WBC"
]

bmCols = [
    "Blast_x",
    "Promyelocyte",
    "Myelocyte",
    "Metamyelocyte",
    "Neutrophils/Band",
    "Monocyte",
    "Eosinophil",
    "Basophil",
    "Erythroid Precursor",
    "Lymphocyte",
    "Plasma Cell",
    "M:E Ratio"
]

def heme_report_gen(prompt, API_KEY, API_ENDPOINT, model=gpt, temperature=1, max_tokens=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    messages = [
        {"role": "system", "content": "You are an expert in the field of hematopathology, able to review patient's clinical lab results and assign a diagnosis category, order immunohistochemical stains and FLOW studies appropriate for that diagnosis."},
        {"role": "user", "content": prompt}
    ]

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def construct_prompt(row, bmCols, cbcCols):
    prompt = """
While acting as a hematopathologist, use the following patient information to fill out the questionnaire. Your responses will be imported into a pandas dataframe so please limit them to short phrases only. You will come up with THREE differential diagnoses for the patient from the list of diagnoses below. List them in decreasing order of likeliness. Then you will decide the type of FLOW most suitable for the patient's case (Myeloid or Lymphoid). Lastly, you will list the immunohistochemical stains you would order to narrow the differential and make a diagnosis. Finally, please provide a one sentence justification of the diagnosis you have chosen.

Possible Diagnoses: Normal, Acute Myeloid Leukemia, Mature B-Cell neoplasm, B-ALL, T-ALL, Acute Lymphoblastic Leukemia, Chronic Lymphocytic Leukemia, Plasma Cell Myeloma, Chronic Myeloid Leukemia, Myelodysplastic syndrome, Myeloproliferative Neoplasm

Questionnaire:
Differential Diagnosis 1:
Differential Diagnosis 2:
Differential Diagnosis 3:
Type of FLOW: (Myeloid or Lymphoid)
IHC: (List the immunohistochemical stains you would order to narrow the differential and make a diagnosis)
Justification: (Please provide a one sentence justification of the diagnosis you have chosen)

Patient Information:

"""
    prompt += f"Patient Age: {row['Age']}\n"
    prompt += "Bone Marrow Differential values:\n"
    bm_values = [f"{col}: {row[col]}" for col in bmCols if col in row]
    prompt += "\n".join(bm_values) + "\n\n"
    prompt += "Peripheral Blood CBC:\n"
    cbc_values = [f"{col}: {row[col]}" for col in cbcCols if col in row]
    prompt += "\n".join(cbc_values) + "\n"

    return prompt

def construct_cat_prompt(row, bmCols, cbcCols):
    prompt = """
While acting as a hematopathologist, use the following patient information to fill out the questionnaire. Your responses will be imported into a Pandas dataframe so please limit them to short phrases only. You will select three three diagnosis categories from the provided list. List them in decreasing order of likeliness. Also include a value (0-100) on how confident you are about the diagnosis. Then you will decide the type of FLOW most suitable for the patient's case (Myeloid, Plasma Cell Myeloma or Lymphoid). Lastly, you will list the immunohistochemical stains you would order to make a diagnosis. Finally, please provide a one sentence justification of the diagnosis you have chosen. Don't be afraid to call a case "Normal".

Possible Diagnostic Categories: "Acute Myeloid Leukemia/Acute Lymphoblastic Leukemia", "Chronic Lymphocytic Leukemia", "Chronic Myeloid Leukemia", "Chronic Myelomonocytic Leukemia", "Mature B-cell Neoplasm", "Myelodysplastic Syndrome", "Plasma Cell Myeloma", or "Normal".

Questionnaire: (Use exact category name without quotes)
Diagnosis Category 1: 
Confidence Score 1:
Diagnosis Category 2:
Confidence Score 2: 
Diagnosis Category 3:
Confidence Score 3:

Type of FLOW: (Myeloid, Plasma Cell Myeloma or Lymphoid)
IHC: (List the immunohistochemical stains you would order to narrow the differential and make a diagnosis)
Justification: (Please provide a one sentence justification of the diagnosis you have chosen)

Patient Information:

"""
    prompt += f"Patient Age: {row['Age']}\n"
    prompt += "Bone Marrow Differential values:\n"
    bm_values = [f"{col}: {row[col]}" for col in bmCols if col in row]
    prompt += "\n".join(bm_values) + "\n\n"
    prompt += "Peripheral Blood CBC:\n"
    cbc_values = [f"{col}: {row[col]}" for col in cbcCols if col in row]
    prompt += "\n".join(cbc_values) + "\n"

    return prompt

def parse_response(text):
    result = {
        'Diagnosis Category 1': None,
        'Confidence Score 1': None,
        'Diagnosis Category 2': None,
        'Confidence Score 2': None,
        'Diagnosis Category 3': None,
        'Confidence Score 3': None,
        'Type of FLOW': None,
        'IHC_GPT': [],
        'Justification': None
    }

    diagnoses = re.findall(r"Diagnosis Category \d+: (.*?)\n", text)
    for i, diagnosis in enumerate(diagnoses, start=1):
        result[f'Diagnosis Category {i}'] = diagnosis.strip()

    confidence_scores = re.findall(r"Confidence Score \d+: (.*?)\n", text)
    for i, confidence_score in enumerate(confidence_scores, start=1):
        result[f'Confidence Score {i}'] = confidence_score.strip()    

    flow_type_match = re.search(r"Type of FLOW: (\w+)", text)
    if flow_type_match:
        result['Type of FLOW'] = flow_type_match.group(1).strip()

    ihc_match = re.search(r"IHC: (.*?)(?=\nJustification:)", text, re.DOTALL)
    if ihc_match:
        ihc_list = ihc_match.group(1).split(',')
        result['IHC_GPT'] = [ihc.strip() for ihc in ihc_list]

    justification = re.search(r"Justification: ([\s\S]+)", text)
    if justification:
        result['Justification'] = justification.group(1).strip()

    return result

def clean_answers(text):
    if text:
        text = text.replace('"', '')
    else:
        text = ""
    return text

def top_k_scores(row):
    ground_truth = row['Dx_Cat']
    top_1 = ground_truth == row['Diagnosis Category 1']
    top_2 = top_1 or ground_truth == row['Diagnosis Category 2']
    top_3 = top_2 or ground_truth == row['Diagnosis Category 3']
    return top_1, top_2, top_3

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config["API_KEY"], config["API_ENDPOINT"]

if __name__ == "__main__":
    API_KEY, API_ENDPOINT = load_config('config.json')  
    df = pd.read_csv("gpt_heme_diff_input_v7.csv")
    #df = df[df['Conclusion'] == 'YES']
    #df = df[df['Dx_Cat'] == 'Normal']

    nrows = input("Enter the number of rows to process or press Enter to process all: ").strip()
    nrows = int(nrows) if nrows.isdigit() else len(df)

    with Progress() as progress:
        task = progress.add_task("[green]Generating Heme Reports...", total=nrows)
        with open("gpt_heme_diff_output_v7_split.txt", "w") as f:
            for index, row in df.iloc[:nrows].iterrows():
                progress.console.log(f"Processing Accession Number: {row['Accession Number']}")
                if row['Conclusion'] != 'YES':
                    progress.advance(task)
                    f.write(f"Accession Number: {row['Accession Number']}\n")
                    f.write(f"Diagnosis: {row['Dx_Cat']}\n")
                    f.write("Row skipped\n")
                    f.write("____________________\n\n")
                    continue
                # if row['Dx_Cat'] != 'Normal':
                #     progress.advance(task)
                #     f.write(f"Accession Number: {row['Accession Number']}\n")
                #     f.write(f"Diagnosis: {row['Dx_Cat']}\n")
                #     f.write("Row skipped due to non-normal diagnosis\n")
                #     f.write("____________________\n\n")
                #     continue
                
                #prompt = construct_prompt(row, bmCols, cbcCols)
                prompt = construct_cat_prompt(row, bmCols, cbcCols)
                
                try:
                    response = heme_report_gen(prompt, API_KEY, API_ENDPOINT)
                    parsed_response = parse_response(response)

                    for key, value in parsed_response.items():
                        if key not in df.columns:
                            df[key] = None
                        df.at[index, key] = str(value)

                except Exception as e:
                    progress.console.log(f"Error processing row {index}: {e}")
                
                f.write(f"Accession Number: {row['Accession Number']}\n")
                f.write(f"Diagnosis: {row['Dx_Cat']}\n")
                f.write("Prompt:\n")
                f.write(prompt + "\n\n\n")
                f.write("Generated Heme Report:\n")
                f.write(response + "\n\n")
                f.write("Parsed Response:\n")
                for key, value in parsed_response.items():
                    f.write(f"{key}: {value}\n")
                f.write("____________________\n\n")
                progress.advance(task)
            
            columns_to_clean = ['Diagnosis Category 1', 'Diagnosis Category 2', 'Diagnosis Category 3']
            df[columns_to_clean] = df[columns_to_clean].applymap(clean_answers)
            df[['Top-1', 'Top-2', 'Top-3']] = df.apply(top_k_scores, axis=1, result_type='expand')
            top_1_accuracy = df['Top-1'].mean()
            top_2_accuracy = df['Top-2'].mean()
            top_3_accuracy = df['Top-3'].mean()
            print(f"Top-1 Accuracy: {top_1_accuracy:.2%}")
            print(f"Top-2 Accuracy: {top_2_accuracy:.2%}")
            print(f"Top-3 Accuracy: {top_3_accuracy:.2%}")
            f.write("_________________________")
            f.write("_________________________\n\n")
            f.write(f"Top-1 Accuracy: {top_1_accuracy:.2%}\n")
            f.write(f"Top-2 Accuracy: {top_2_accuracy:.2%}\n")
            f.write(f"Top-3 Accuracy: {top_3_accuracy:.2%}\n")

    df.to_csv("gpt_heme_diff_output_v7_split.csv", index=False)
