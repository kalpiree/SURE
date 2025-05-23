import pandas as pd

file_path = "outputs/fmlp_runs/goodreads/alpha_007/detailed_snapshots.csv"  # Change path if needed
df = pd.read_csv(file_path)

print("Columns in the file:\n")
for col in df.columns:
    print(col)
