import pandas as pd
import io
import os

def prepare_data():
    filename = 'KDDTest+.arff'
    
    if not os.path.exists(filename):
        print(f"❌ ERROR: Place '{filename}' in this folder first!")
        return

    print("--- Sentinel AI: Initializing Data Clean-up ---")
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_start_idx = 0
    column_names = []
    for i, line in enumerate(lines):
        if line.lower().startswith('@attribute'):
            col_name = line.split()[1].strip("'")
            column_names.append(col_name)
        if line.lower().startswith('@data'):
            data_start_idx = i + 1
            break

    # Clean data content: removes quotes and spaces that cause ValueErrors
    data_content = "".join(lines[data_start_idx:]).replace("'", "").replace(" ", "")
    
    df = pd.read_csv(io.StringIO(data_content), names=column_names, header=None)
    
    # Standardize the last column (the class)
    last_col = df.columns[-1]
    df[last_col] = df[last_col].apply(lambda x: 'normal' if x == 'normal' else 'anomaly')
    
    # Save the master file
    df.to_csv('cleaned_sentinel_data.csv', index=False)
    print("✅ Success: 'cleaned_sentinel_data.csv' created.")
    print(df[last_col].value_counts())

if __name__ == "__main__":
    prepare_data()