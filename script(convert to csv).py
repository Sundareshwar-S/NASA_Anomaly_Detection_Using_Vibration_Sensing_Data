import os
import pandas as pd

# ----------------- 📁 Step 1: Set Up Paths -----------------
# The source directory with your raw text files
input_dir = '/home/sundaershwar/Downloads/archive/2nd_test/2nd_test/'

# A new directory to save the output CSV files
output_dir = 'output_csvs'

# ----------------- ⚙️ Step 2: Create Output Directory -----------------
# This creates the folder if it doesn't already exist.
os.makedirs(output_dir, exist_ok=True)

# ----------------- 💾 Step 3: Process and Save Files Separately -----------------
if not os.path.isdir(input_dir):
    print(f"Error: The directory '{input_dir}' does not exist.")
else:
    all_files = sorted(os.listdir(input_dir))
    
    # Limit the list to the first 4 files
    files_to_process = all_files[:4]
    
    print(f"Converting {len(files_to_process)} files to separate CSVs...")

    # Loop through each of the 4 files
    for filename in files_to_process:
        # Define the full path for the input file
        input_file_path = os.path.join(input_dir, filename)
        
        # Create the new CSV filename using the original timestamp
        # Example: '2004.02.12.10.32.39' becomes '2004.02.12.10.32.39.csv'
        output_filename = f"{filename}.csv"
        output_file_path = os.path.join(output_dir, output_filename)
        
        # Load the raw text file
        df = pd.read_csv(input_file_path, sep='\t', header=None, names=['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4'])
        
        # Save the data to its own new CSV file
        df.to_csv(output_file_path, index=False)
        
        print(f"  -> Saved {output_filename}")

    print(f"\n✅ Success! All files converted.")
    print(f"Check the '{output_dir}' folder.")