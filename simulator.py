import os
import datetime
import copy
import pandas as pd

# Import configuration modules
from config_gui import get_user_parameters
from config_auto import generate_random_parameters

# List of algorithms to test
import algo_pomcp
import algo_auction
import algo_greedy

ALGORITHMS = [
    ("DEC-POMCP", algo_pomcp),
    ("AUCTION", algo_auction),
    ("GREEDY", algo_greedy)
]

def main():
    print("=== MULTI-ALGORITHM SIMULATOR ===")
    try:
        num_scenarios = int(input("How many scenarios do you want to simulate? "))
        print("\nHow do you want to configure the scenarios?")
        print("1. Manually (via GUI)")
        print("2. Automatic Procedural Generation (Random heuristics)")
        scelta_config = input("Choice (1 or 2): ")
    except ValueError:
        print("Error: Please enter a valid integer.")
        return

    # =========================================================================
    # FOLDER CREATION FOR CURRENT SESSION RESULTS
    # =========================================================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    now = datetime.datetime.now()
    folder_name = now.strftime("Sim_%Y-%m-%d_%H-%M-%S")
    session_folder = os.path.join(base_dir, folder_name)
    os.makedirs(session_folder, exist_ok=True)
    print(f"\n[+] Session folder created: {session_folder}")

    # =========================================================================
    # 1 - PARAMETER CONFIGURATION FOR ALL SCENARIOS
    # =========================================================================
    all_scenarios_params = []
    
    for i in range(1, num_scenarios + 1):
        if scelta_config == '1':
            print(f"\n{'='*50}")
            print(f" STARTING SCENARIO CONFIGURATION {i}/{num_scenarios}")
            print(f"{'='*50}")
            params = get_user_parameters(scenario_idx=i, save_folder=session_folder)
        else:
            params = generate_random_parameters(scenario_idx=i)
            
        all_scenarios_params.append(params)
        
    print("\n" + "="*50)
    print(" STARTING SIMULATIONS FOR ALL SCENARIOS ")
    print("="*50)

    # =========================================================================
    # 2 - SIMULATION EXECUTION FOR ALL SCENARIOS AND ALGORITHMS
    # =========================================================================
    results_data = []

    # Loop for extracting parameters for each scenario
    for params in all_scenarios_params:
        scenario_idx = params['scenario_idx']
        
        # Data for excel report
        map_type_str = "Uniform" if params['map_type'] == 1 else "Multi-Gaussian"
        has_traces = "Yes" if len(params.get('traces', [])) > 0 else "No"
        
        scenario_record = {
            "Scenario": scenario_idx,
            "N. Drones": params['num_drones'],
            "Map Type": map_type_str,
            "Traces": has_traces
        }

        for algo_name, algo_module in ALGORITHMS:
            print(f"\n Executing: {algo_name} (Scenario {scenario_idx})")
            
            # Deep copy to prevent overriding parameters between runs
            current_params = copy.deepcopy(params)
            current_params['algo_name'] = algo_name
            current_params['save_folder'] = session_folder
            
            # Simulation execution: returns steps taken and list of entropy values
            steps_taken, entropy_list = algo_module.run_simulation(current_params)
            
            if steps_taken == -1:
                print(f"Simulation {algo_name} stopped by user or encountered an error.")
                scenario_record[f"Step {algo_name}"] = "Interrupted/Error"
                scenario_record[f"Entropy {algo_name}"] = []
            else:
                scenario_record[f"Step {algo_name}"] = steps_taken
                scenario_record[f"Entropy {algo_name}"] = entropy_list

        results_data.append(scenario_record)

    # =========================================================================
    # 3 - EXCEL REPORT GENERATION
    # =========================================================================
    generate_excel_report(results_data, session_folder)


def generate_excel_report(data, session_folder):
    if not data:
        print("No data to export.")
        return

    df = pd.DataFrame(data)
    
    # Determine the winners for each scenario (handling ties)
    algo_names = [name for name, _ in ALGORITHMS]
    step_columns = [f"Step {name}" for name in algo_names]
    
    winners_per_scenario = []
    # Initialize win counts to 0 for all algorithms
    win_counts_dict = {name: 0 for name in algo_names}
    
    for index, row in df.iterrows():
        # Convert steps to numeric, ignoring "Interrupted/Error" strings
        valid_steps = pd.to_numeric(row[step_columns], errors='coerce')
        
        if valid_steps.notna().any():
            # Find the absolute minimum step value
            min_step = valid_steps.min()
            
            # Get ALL columns that match this minimum value (handling ties)
            best_cols = valid_steps[valid_steps == min_step].index.tolist()
            
            # Clean the names (remove "Step ")
            scenario_winners = [col.replace("Step ", "") for col in best_cols]
            
            # Add +1 win to each algorithm that tied
            for w in scenario_winners:
                win_counts_dict[w] += 1
                
            # Join names for the Excel column (e.g., "AUCTION, GREEDY")
            winners_per_scenario.append(", ".join(scenario_winners))
        else:
            winners_per_scenario.append("None")
            
    df["Winner"] = winners_per_scenario
    
    # Convert the dictionary back to a pandas Series to keep compatibility with the chart code
    win_counts = pd.Series(win_counts_dict)
    # Remove algorithms with 0 wins to keep the pie chart clean
    win_counts = win_counts[win_counts > 0]

    # Separate entropy columns from summary columns
    entropy_columns = [col for col in df.columns if col.startswith("Entropy ")]
    df_summary = df.drop(columns=entropy_columns)

    # Creation of Excel report
    filename = os.path.join(session_folder, "Report_Simulations.xlsx")
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book
    
    # --- SHEET 1: SUMMARY AND PIE CHART ---
    df_summary.to_excel(writer, sheet_name='Results', index=False)
    worksheet_results = writer.sheets['Results']
    
    # Auto-adjust column widths
    for i, col in enumerate(df_summary.columns):
        column_len = max(df_summary[col].astype(str).map(len).max(), len(col)) + 2
        worksheet_results.set_column(i, i, column_len)

    # Add pie chart if there are winners
    if not win_counts.empty and win_counts.index[0] != "None":
        chart_data_start_row = len(df_summary) + 5
        worksheet_results.write(chart_data_start_row, 0, "Algorithm")
        worksheet_results.write(chart_data_start_row, 1, "Wins")
        
        for i, (algo, count) in enumerate(win_counts.items()):
            worksheet_results.write(chart_data_start_row + 1 + i, 0, algo)
            worksheet_results.write(chart_data_start_row + 1 + i, 1, count)

        pie_chart = workbook.add_chart({'type': 'pie'})
        pie_chart.add_series({
            'name': 'Win Percentage',
            'categories': ['Results', chart_data_start_row + 1, 0, chart_data_start_row + len(win_counts), 0],
            'values':     ['Results', chart_data_start_row + 1, 1, chart_data_start_row + len(win_counts), 1],
            'data_labels': {'percentage': True, 'category': True, 'separator': '\n'}
        })
        pie_chart.set_title({'name': 'Algorithms Win Distribution'})
        worksheet_results.insert_chart('H2', pie_chart)

    # --- SHEET 2: ENTROPY DATA AND IMPROVED LINE CHARTS ---
    worksheet_entropy = workbook.add_worksheet('Entropy_Graphs')
    
    col_offset = 0 
    chart_row_offset = 2 
    
    for index, row in df.iterrows():
        scenario_num = row['Scenario']
        
        # Extract entropy data into a temporary DataFrame (aligns arrays of different lengths with empty cells)
        dict_scenario_entropies = {}
        for col in entropy_columns:
            algo_name = col.replace("Entropy ", "")
            dict_scenario_entropies[algo_name] = pd.Series(row[col]) 
            
        df_ent = pd.DataFrame(dict_scenario_entropies)
        
        # Write raw data to the sheet
        worksheet_entropy.write(0, col_offset, f"Data Scenario {scenario_num}")
        df_ent.to_excel(writer, sheet_name='Entropy_Graphs', startrow=1, startcol=col_offset, index=False)
        
        # Calculate min/max for Y-axis zoom
        min_entropy = df_ent.min().min()
        max_entropy = df_ent.max().max()
        margin = (max_entropy - min_entropy) * 0.05
        if pd.isna(margin) or margin == 0:
            margin = 0.5
            
        y_min = max(0, min_entropy - margin)
        y_max = max_entropy + margin

        line_chart = workbook.add_chart({'type': 'line'})
        
        line_styles = ['solid', 'dash', 'dot', 'dash_dot']
        markers = ['circle', 'square', 'triangle', 'diamond']
        
        # Add a series for each tested algorithm
        for i, col_name in enumerate(df_ent.columns):
            num_rows = len(df_ent[col_name].dropna()) 
            
            line_chart.add_series({
                'name':       ['Entropy_Graphs', 1, col_offset + i],
                'values':     ['Entropy_Graphs', 2, col_offset + i, 1 + num_rows, col_offset + i],
                'marker':     {'type': markers[i % len(markers)], 'size': 5},
                'line':       {'width': 2.25, 'dash_type': line_styles[i % len(line_styles)]}
            })
            
        line_chart.set_title({'name': f'Entropy Trend - Scenario {scenario_num}'})
        line_chart.set_x_axis({
            'name': 'Steps', 
            'major_gridlines': {'visible': True}
        })
        line_chart.set_y_axis({
            'name': 'Entropy (Bits)',
            'min': y_min,
            'max': y_max,
            'num_format': '0.000'
        })
        line_chart.set_size({'width': 750, 'height': 400})
        
        # Insert the chart next to the raw data
        worksheet_entropy.insert_chart(chart_row_offset, col_offset + len(df_ent.columns) + 2, line_chart)
        
        # Update offsets for the next scenario
        chart_row_offset += 18 
        col_offset += len(df_ent.columns) + 1 

    writer.close()
    print(f"\n{'='*50}")
    print(f"Simulation completed! Data saved in folder:\n{session_folder}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()