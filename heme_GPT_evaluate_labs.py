import requests
import json
import pandas as pd
import re
from rich.progress import Progress

#gpt = "gpt-3.5-turbo"
gpt = "gpt-4"

def heme_evaluate_labs(prompt, API_KEY, API_ENDPOINT, model=gpt, temperature=1, max_tokens=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    messages = [
        {"role": "system", "content": "You are an expert in field of hematopathology and especially at analyzing bone marrow and peripheral blood lab values."},
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
While acting as a hematopathologist, analyze the following patient information. Your responses will be imported into a pandas dataframe so please stick to the answer format given below. You will look at the patient's diagnosis, then look at the given lab values for Bone Marrow Differential as well as the Peripheral Blood CBC values. Your job is to determine if the characteristic lab abnormality that is usually found with that diagnosis, is present in the given lab values or not. Finally, you will provide a short comment describing your reasoning. The comment should include which lab abnormality you expected, whether it was found or not based on the reference values given in literature for that lab value. 

Please strictly adhere to the following format for your response so that it can be easily parsed and imported into a pandas dataframe:

Conclusion: Was the characteristic lab abnormality present or not? [YES or NO]
Comment: [Your comment here]

Patient Information:
"""
    prompt += f"Patient Age: {row['Age']}\n"
    prompt += "Patient's Diagnosis:\n"
    prompt += f"{row['Dx_Cat']}\n\n"
    prompt += "Bone Marrow Differential values:\n"
    bm_values = [f"{col}: {row[col]}" for col in bmCols if col in row]
    prompt += "\n".join(bm_values) + "\n\n"
    prompt += "Peripheral Blood CBC:\n"
    cbc_values = [f"{col}: {row[col]}" for col in cbcCols if col in row]
    prompt += "\n".join(cbc_values) + "\n"

    return prompt

CONCLUSION_PATTERN = re.compile(r"Conclusion:\s*(YES|NO)")
COMMENT_PATTERN = re.compile(r"Comment:([\s\S]+)")

def parse_response(text):
    try:
        conclusion_search = CONCLUSION_PATTERN.search(text)
        if conclusion_search:
            conclusion = conclusion_search.group(1).strip()
        else:
            conclusion = "No Conclusion" 

        comment_search = COMMENT_PATTERN.search(text)
        if comment_search:
            comment = comment_search.group(1).strip()
        else:
            comment = "No Comment"  

    except AttributeError:
        conclusion = "Error"  
        comment = "Pattern not found in response."

    return conclusion, comment

def col_to_text(row, columns):
    values = [f"{col} - {str(row[col])}" for col in columns]
    return "\n".join(values)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    API_KEY = config["API_KEY"]
    API_ENDPOINT = config["API_ENDPOINT"]
    return API_KEY, API_ENDPOINT

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

if __name__ == "__main__":
    API_KEY, API_ENDPOINT = load_config("config.json")
    df = pd.read_csv("updated_case_list_with_cbc_split.csv")
    for column in ['Conclusion', 'Comment']:
        if column not in df.columns:
            df[column] = ''

    nrows = input("Enter the number of rows to process or press Enter to process all: ").strip()
    nrows = int(nrows) if nrows.isdigit() else len(df)

    with Progress() as progress:
        task = progress.add_task("[blue]Evaluating CBC and BM Differential...", total=nrows)

        with open("lab_eval_output_v7_split.txt", "w") as f:
            for index, row in df.iloc[:nrows].iterrows():
                # dx_filter = ['Chronic Myeloid Leukemia/Chronic Myelomonocytic Leukemia','Mature B-cell Neoplasm', 'Chronic Myeloid Leukemia', 'Chronic Myelomonocytic Leukemia', 'Normal']
                # if row['Dx_Cat'] not in dx_filter:
                #     progress.advance(task)
                #     f.write(f"Accession Number: {row['Accession Number']}\n")
                #     f.write(f"Diagnosis: {row['Dx_Cat']}\n")
                #     f.write("Row skipped\n")
                #     f.write("____________________\n\n")
                #     continue
                prompt = construct_prompt(row, bmCols, cbcCols) 
                f.write(f"Accession Number: {row['Accession Number']}\n")
                f.write("Prompt:\n")
                f.write(prompt + "\n\n")

                try:
                    response = heme_evaluate_labs(prompt, API_KEY, API_ENDPOINT)
                    conclusion, comment = parse_response(response)
                    f.write("Response:\n")
                    f.write(response + "\n\n")
                    f.write("Conclusion:\n")
                    f.write(conclusion + "\n\n")
                    f.write("Comment:\n")
                    f.write(comment + "\n\n")
                    f.write("____________________\n\n")
                    df.at[index, 'Conclusion'] = conclusion
                    df.at[index, 'Comment'] = comment

                except Exception as e:
                    error_message = f"Error processing row {index}: {e}"
                    f.write(error_message + "\n\n")
                    print(error_message) 
                progress.update(task, advance=1)
        
        df.to_csv("lab_eval_output_v7_split.csv", index=False)
            
