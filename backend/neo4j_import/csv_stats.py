import os
import pandas as pd

FOLDER = "neo4j_import"   # <-- update path if needed

def inspect_csv_files(folder_path):
    print(f"\nInspecting CSV files in: {folder_path}\n" + "-"*80)
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            
            try:
                df = pd.read_csv(filepath)
                num_rows = len(df)
                
                print(f"File: {filename}")
                print(f"   ➤ Rows: {num_rows}")
                print(f"   ➤ Columns: {list(df.columns)}")
                
                # print top 3 rows
                print("\n   ➤ Top 3 Records:")
                print(df.head(3).to_string(index=False))
                
                print("-"*80)
            
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                print("-"*80)

if __name__ == "__main__":
    inspect_csv_files(FOLDER)