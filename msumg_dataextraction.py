import pandas as pd
import os
import shutil
import pprint


# Paths to CSV files for each dataset
listofpaths = {
    "MS-UMG": [
        r"\path\to\MS-UMG\id\2020\2020_regular.csv",
        r"\path\to\MS-UMG\id\2021\2021_regular.csv"
    ]
}

# Define the drug lists for each DRIAMS dataset
drug_columns = {
    "MS-UMG": ["Amoxicillin", "Ampicillin"]
}

# Define the output directories for ampicillin susceptibility status
output_dirs = {
    "MS-UMG": {
        "ECFMAR": r"path\to\your\outputdir\MS-UMG_ECFMAR_preprocessed_novre_regular",
        "ECFMAS": r"path\to\your\outputdir\\MS-UMG_ECFMAS_preprocessed_novre_regular"
    }
}

# Ensure output directories exist
for dataset in output_dirs:
    os.makedirs(output_dirs[dataset]["ECFMAR"], exist_ok=True)
    os.makedirs(output_dirs[dataset]["ECFMAS"], exist_ok=True)

# Function to process each file
def process_file(path, drug_columns):
    df = pd.read_csv(path, low_memory=False, na_values=["-", ""], dtype=str)
    enterococcus_subset = df[df['species'] == "Enterococcus faecium"]

    results = []
    # Check for the presence of columns and handle rows accordingly
    if all(col in enterococcus_subset.columns for col in drug_columns):
        for index, row in enterococcus_subset.iterrows():
            drug_values = [row[col] for col in drug_columns]
            vancomycin_value = row['Vancomycin']

            if vancomycin_value == "S":

                if all(val == "R" for val in drug_values):
                    results.append(("ECFMAR", row))
                elif any(val == "R" and pd.isna(val2) for val, val2 in zip(drug_values, drug_values[1:] + [drug_values[0]])):
                    results.append(("ECFMAR", row))
                elif all(val == "S" for val in drug_values):
                    results.append(("ECFMAS", row))
                elif any(val == "S" and pd.isna(val2) for val, val2 in zip(drug_values, drug_values[1:] + [drug_values[0]])):
                    results.append(("ECFMAS", row))
                elif any(val == "R" and val2 == "S" or val == "S" and val2 == "R" for val, val2 in zip(drug_values, drug_values[1:] + [drug_values[0]])):
                    results.append(("errors", row))
                elif all(pd.isna(val) for val in drug_values):
                    results.append(("NaN", row))
                else:
                    results.append(("errors", row))
    else:
        for col in drug_columns:
            if col not in enterococcus_subset.columns:
                print(f"Warning: '{col}' column missing in {path}")
    return results

# Function to process datasets
def process_datasets(listofpaths, drug_columns, output_dirs):
    for dataset, paths in listofpaths.items():
        print(f"Processing {dataset}...")

        ECFMAR = pd.DataFrame()
        ECFMAS = pd.DataFrame()
        errors = pd.DataFrame()
        NaN = pd.DataFrame()

        for path in paths:
            results = process_file(path, drug_columns[dataset])
            for result in results:
                category, row = result
                if category == "ECFMAR":
                    ECFMAR = pd.concat([ECFMAR, pd.DataFrame([row])], ignore_index=True)
                elif category == "ECFMAS":
                    ECFMAS = pd.concat([ECFMAS, pd.DataFrame([row])], ignore_index=True)
                elif category == "errors":
                    errors = pd.concat([errors, pd.DataFrame([row])], ignore_index=True)
                elif category == "NaN":
                    NaN = pd.concat([NaN, pd.DataFrame([row])], ignore_index=True)
            print(f"Result {path}: {len(results)}")
            print(f"ECFMAR {path}: {len(ECFMAR)}")
            print(f"ECFMAS {path}: {len(ECFMAS)}")
            print(f"errors {path}: {len(errors)}")
            print(f"NaN {path}: {len(NaN)}")

        # Store "code" column values to lists
        ECFMAR_codes = ECFMAR["code"].tolist()
        ECFMAS_codes = ECFMAS["code"].tolist()

        # Source directory to walk through
        source_dir = r"path\to\MS-UMG\\preprocessed"

        # Copy files for ECFMAR and ECFMAS codes
        no_match_ECFMAR, multiple_matches_ECFMAR = copy_files(ECFMAR_codes, source_dir, output_dirs[dataset]["ECFMAR"])
        no_match_ECFMAS, multiple_matches_ECFMAS = copy_files(ECFMAS_codes, source_dir, output_dirs[dataset]["ECFMAS"])

        # Flag any codes with no matches or multiple matches
        if no_match_ECFMAR:
            print(source_dir)
            print(f"No matches for ECFMAR codes: {len(no_match_ECFMAR)}")
        if multiple_matches_ECFMAR:
            print(f"Multiple matches for ECFMAR codes: {len(multiple_matches_ECFMAR)}")
            pprint.pprint(multiple_matches_ECFMAR)
        if no_match_ECFMAS:
            print(f"No matches for ECFMAS codes: {len(no_match_ECFMAS)}")
        if multiple_matches_ECFMAS:
            print(f"Multiple matches for ECFMAS codes: {len(multiple_matches_ECFMAS)}")

        # Save the resulting DataFrames to new CSV files
        output_dir = r"path\to\your\dataframes\MS-UMG_preprocessed_df_regular"
        os.makedirs(output_dir, exist_ok=True)
        
        ECFMAR.to_csv(os.path.join(output_dir, f"ECFMAR_novre{dataset}.csv"), index=False)
        ECFMAS.to_csv(os.path.join(output_dir, f"ECFMAS_novre{dataset}.csv"), index=False)
        errors.to_csv(os.path.join(output_dir, f"errors_novre{dataset}.csv"), index=False)
        NaN.to_csv(os.path.join(output_dir, f"NaN_novre{dataset}.csv"), index=False)

# Function to copy files based on code list
def copy_files(code_list, source_dir, target_dir):
    no_match = []
    multiple_matches = []
    for code in code_list:
        matches = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".txt") and file.startswith(code):
                    matches.append(os.path.join(root, file))
        if len(matches) == 1:
            shutil.copy(matches[0], target_dir)
        elif len(matches) == 0:
            no_match.append(code)
        else:
            multiple_matches.append(code)
            shutil.copy(matches[0], target_dir)
            pprint.pprint(matches)
    return no_match, multiple_matches

# Process the datasets
process_datasets(listofpaths, drug_columns, output_dirs)
