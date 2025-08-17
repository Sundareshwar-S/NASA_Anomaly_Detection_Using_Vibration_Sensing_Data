from flask import Flask, render_template, request, flash, redirect, url_for
import os
import io
from contextlib import redirect_stdout

# CORRECTED IMPORT: Using the correct function name 'load_data_from_mysql'
from pipeline import PipelineManager, load_data_from_mysql

app = Flask(__name__)
app.secret_key = 'supersecretkey' 

# --- Configuration ---
STATIC_DIR = 'static'
PLOT_DIR = os.path.join(STATIC_DIR, 'plots')

@app.route('/')
def index():
    """Renders the home page with the database connection form."""
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_analysis():
    """
    This is the main function that runs when the form is submitted.
    It executes the entire data science pipeline.
    """
    # Get database credentials from the web form
    db_creds = {
        "user": request.form.get('db_user'),
        "password": request.form.get('db_password'),
        "host": request.form.get('db_host'),
        "database": request.form.get('db_name')
    }
    num_tables = request.form.get('num_tables', default=1, type=int)
    bearings_input = request.form.get('bearings_to_analyze', '1')
    target_columns = [f"B{s.strip()}_x" for s in bearings_input.split(',')]

    log_stream = io.StringIO()
    with redirect_stdout(log_stream):
        try:
            print("--- Starting Intelligent Pipeline ---")
            
            # 1. Load Data from MySQL
            # CORRECTED FUNCTION CALL: Using the correct function name
            main_df = load_data_from_mysql(db_creds, num_tables_to_load=num_tables)
            
            if main_df is None or main_df.empty:
                flash("Error: Could not load data. Please check DB credentials.", "error")
                return redirect(url_for('index'))

            all_results = []
            # 2. Loop and run the pipeline for each selected bearing
            for target_col in target_columns:
                print(f"\n\n========================================================")
                print(f"  STARTING ANALYSIS FOR TARGET: {target_col}  ")
                print(f"========================================================")
                
                if target_col not in main_df.columns:
                    print(f"Warning: Column '{target_col}' not found in data. Skipping.")
                    continue

                pipeline = PipelineManager(main_df, target_column=target_col) 
                final_report, plot_paths = pipeline.execute_pipeline(plot_dir=PLOT_DIR)
                all_results.append({
                    'target': target_col,
                    'report': final_report,
                    'plots': plot_paths
                })
            
            print("\n--- All Analyses Finished Successfully ---")

        except Exception as e:
            print(f"\n--- An Error Occurred ---")
            print(f"Error details: {e}")
            all_results = [{'target': 'Error', 'report': 'Pipeline failed to run. Check logs for details.', 'plots': []}]

    execution_log = log_stream.getvalue()
    
    return render_template('results.html', 
                           results=all_results,
                           logs=execution_log)

if __name__ == '__main__':
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    app.run(debug=True)