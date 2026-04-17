import os
import datetime
import copy
import pandas as pd
import numpy as np


# Import configuration modules
from config_gui import (DEFAULT_CONFIG, get_user_parameters, generate_random_parameters)

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
        num_scenarios = int(input("Number of scenarios to simulate: "))
        print("\nConfiguration Method:")
        print("1. Manual Configuration")
        print("2. Automatic Configuration")
        scelta_config = input("Choice (1 or 2): ")

        num_alphas = int(input("How many different reward_alpha values to test? "))
        alpha_values = []
        for i in range(num_alphas):
            val = float(input(f"Enter the value for reward_alpha #{i+1}: "))
            alpha_values.append(val)

    except ValueError:
        print("Error: Please enter a valid number.")
        return

    # Folder creation
    base_dir = os.path.dirname(os.path.abspath(__file__))
    now = datetime.datetime.now()
    global_timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    folder_name = f"Sim_{global_timestamp}"
    session_folder = os.path.join(base_dir, folder_name)
    os.makedirs(session_folder, exist_ok=True)
    print(f"\n Session folder created: {session_folder}")

    # =========================================================================
    # 1 - PARAMETER CONFIGURATION FOR ALL SCENARIOS 
    # =========================================================================
    all_scenarios_params = []
    
    for i in range(1, num_scenarios + 1):
        if scelta_config == '1':
            print(f"\n{'='*50}")
            print(f" STARTING SCENARIO CONFIGURATION {i}/{num_scenarios}")
            print(f"{'='*50}")
            params = get_user_parameters(scenario_idx=i)
        else:
            params = generate_random_parameters(scenario_idx=i)
            
        all_scenarios_params.append(params)
        
    print("\n" + "="*50)
    print(" STARTING SIMULATIONS FOR ALL SCENARIOS ")
    print("="*50)

    # =========================================================================
    # 2 - ITERATIONS OVER ALPHA VALUES, SCENARIOS AND ALGORITHMS
    # =========================================================================
    
    # Loop over different reward_alpha values
    for current_alpha in alpha_values:
        print(f"\n\n{'#'*60}")
        print(f" STARTING SIMULATIONS WITH REWARD_ALPHA = {current_alpha} ")
        print(f"{'#'*60}")

        results_data = []

        # Loop for extracting parameters for each scenario
        for params in all_scenarios_params:
            scenario_idx = params['scenario_idx']
            
            scenario_record = {
                "Scenario": scenario_idx,
                "Drone Pos": str(params.get('drone_positions', [])),
                "Target Pos": str(params.get('target_pos', '')),
                "Num Peaks": len(params.get('peaks', [])),
                "Peak Means": str([p['mean'] for p in params.get('peaks', [])]),
                "Peak Vars": str([p['cov'] for p in params.get('peaks', [])]),
                "Obstacles": str(params.get('obstacles', [])),
                "Step POMCP": None,
                "Step AUCTION": None,
                "Step GREEDY": None,
                "pomcp_metrics": []
            }

            for algo_name, algo_module in ALGORITHMS:
                print(f"\n Executing: {algo_name} (Scenario {scenario_idx} | Alpha {current_alpha})")
                
                current_params = copy.deepcopy(params)
                current_params['algo_name'] = algo_name
                current_params['save_folder'] = session_folder
                current_params['reward_alpha'] = current_alpha      # Set current alpha in parameters
                
                # Simulation execution: returns steps taken and metrics
                res = algo_module.run_simulation(current_params)
                steps_taken = res[0]
                
                # Manual interruptions 
                if steps_taken == -1:
                    print(f"Simulation {algo_name} stopped by user.")
                    steps_taken = float('inf') 
                
                if algo_name == "DEC-POMCP":
                    scenario_record["Step POMCP"] = steps_taken
                    scenario_record["pomcp_metrics"] = res[2] if len(res) > 2 else []
                elif algo_name == "AUCTION":
                    scenario_record["Step AUCTION"] = steps_taken
                elif algo_name == "GREEDY":
                    scenario_record["Step GREEDY"] = steps_taken

            results_data.append(scenario_record)

        # Call to the function to generate Excel report
        generate_excel_report(results_data, session_folder, global_timestamp, current_alpha)


# Function to generate Excel report 
def generate_excel_report(data, session_folder, timestamp, alpha_reward):
    if not data:
        print("No data to export.")
        return
    
    filename = os.path.join(session_folder, f"{timestamp}_{alpha_reward}.xlsx")
    
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book

    # SHEET 1: SCENARIO PARAMETERS
    df_params = pd.DataFrame(data)[["Scenario", "Drone Pos", "Target Pos", "Num Peaks", "Peak Means", "Peak Vars", "Obstacles"]]
    df_params["Num Steps Map"] = DEFAULT_CONFIG['map_size'] 
    df_params.to_excel(writer, sheet_name='Parametri', index=False)
    
    worksheet_param = writer.sheets['Parametri']
    for i, col in enumerate(df_params.columns):
        column_len = max(df_params[col].astype(str).map(len).max(), len(col)) + 2
        worksheet_param.set_column(i, i, column_len)

    # SHEET 2: ALGORITHM PERFORMANCE AND WINNERS
    performance_records = []
    win_counts = {"DEC-POMCP": 0, "AUCTION": 0, "GREEDY": 0}
    
    for row in data:
        steps_dict = {
            "DEC-POMCP": row["Step POMCP"],
            "AUCTION": row["Step AUCTION"],
            "GREEDY": row["Step GREEDY"]
        }
        
        valid_steps = {k: v for k, v in steps_dict.items() if v != float('inf') and v is not None}
        
        if valid_steps:
            min_step = min(valid_steps.values())
            winners = [k for k, v in valid_steps.items() if v == min_step]
            for w in winners:
                win_counts[w] += 1
            winner_str = ", ".join(winners)
        else:
            winner_str = "Nessuno (Interrotto)"

        rec = {"Scenario": row["Scenario"]}
        rec.update({k: (v if v != float('inf') else "Interrotto") for k,v in steps_dict.items()})
        rec["Vincitori"] = winner_str
        performance_records.append(rec)

    total_row = {"Scenario": "TOTALE VITTORIE"}
    total_row.update(win_counts)
    total_row["Vincitori"] = ""
    performance_records.append(total_row)

    df_perf = pd.DataFrame(performance_records)
    df_perf.to_excel(writer, sheet_name='Performance', index=False)
    worksheet_perf = writer.sheets['Performance']
    for i, col in enumerate(df_perf.columns):
        worksheet_perf.set_column(i, i, max(len(col)+2, 15))

    # SHEET 3: POMCP METRICS FOR DRONE 1
    metrics_records = []
    for row in data:
        all_metrics = row["pomcp_metrics"]
        # Recupera solo i dati del Drone 1 (escludendo i None del TSP)
        if all_metrics and 1 in all_metrics:
            m_list = [m for m in all_metrics[1] if m is not None]
            if m_list:
                iters = [m['iterations'] for m in m_list]
                depths = [m['depth'] for m in m_list]
                nodes = [m['nodes'] for m in m_list]
                
                metrics_records.append({
                    "Scenario": row["Scenario"],
                    "Iterazioni Min": np.min(iters),
                    "Iterazioni Max": np.max(iters),
                    "Iterazioni Medie": round(np.mean(iters), 2),
                    "Depth Min": np.min(depths),
                    "Depth Max": np.max(depths),
                    "Depth Media": round(np.mean(depths), 2),
                    "Nodi Min": np.min(nodes),
                    "Nodi Max": np.max(nodes),
                    "Nodi Medi": round(np.mean(nodes), 2)
                })
            else:
                metrics_records.append({"Scenario": row["Scenario"]})
        else:
            metrics_records.append({"Scenario": row["Scenario"]})
            
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_excel(writer, sheet_name='Metrics_Drone1', index=False)
    worksheet_met = writer.sheets['Metrics_Drone1']
    for i, col in enumerate(df_metrics.columns):
        worksheet_met.set_column(i, i, max(len(col)+2, 12))

    # Helper function to create diagrams in excel
    def create_spaced_chart_sheet(sheet_name, data, metric_key, chart_title, y_axis_name):
        ws = workbook.add_worksheet(sheet_name)
        row_offset = 0
        
        for row in data:
            scenario = row["Scenario"]
            metrics_dict = row["pomcp_metrics"]
            
            if not metrics_dict:
                continue
                
            # Find max number of steps for the current scenario across all drones to set chart range
            max_steps = max((len(m_list) for m_list in metrics_dict.values()), default=0)
            if max_steps == 0:
                continue
                
            # Write scenario title and step headers
            ws.write(row_offset, 0, f"Scenario {scenario}")
            for step_idx in range(max_steps):
                ws.write(row_offset, step_idx + 1, f"Step {step_idx+1}")
            
            # Build chart 
            chart = workbook.add_chart({'type': 'line'})
            
            current_row = row_offset + 1
            
            # Write data for each drone and add series to the chart
            for drone_id, m_list in metrics_dict.items():
                ws.write(current_row, 0, f"Drone {drone_id}")
                
                vals = [m.get(metric_key) if m is not None else None for m in m_list]
                
                for col_idx, val in enumerate(vals):
                    if val is not None:
                        ws.write(current_row, col_idx + 1, round(val, 2))
                
                # Add series to the chart for this drone
                chart.add_series({
                    'name': f'Drone {drone_id}',
                    'categories': [sheet_name, row_offset, 1, row_offset, max_steps],
                    'values': [sheet_name, current_row, 1, current_row, max_steps],
                    'marker': {'type': 'circle', 'size': 4},
                    'line': {'width': 1.5}
                })
                
                current_row += 1
            
            # Chart formatting
            chart.set_title({'name': f'{chart_title} - Scenario {scenario}'})
            chart.set_x_axis({'name': 'Time Steps (Global)'})
            chart.set_y_axis({'name': y_axis_name})
            chart.set_size({'width': 700, 'height': 350})
            
            # Chart positioning 
            ws.insert_chart(current_row + 1, 1, chart)
            
            # Add space
            row_offset = current_row + 22

    # SHEET 4: EXPLORATION RATIO 
    create_spaced_chart_sheet('Expl_Ratio', data, 'expl_ratio', 'Exploration vs Exploitation Ratio', 'Ratio %')

    # SHEET 5: ROOT ACTION FLIPS 
    create_spaced_chart_sheet('Root_Flips', data, 'flips', 'Variazioni Azione al Root', 'Numero Flips')

    writer.close()
    print(f"\n{'='*50}")
    print(f"Simulation completed! Report Excel saved as:\n{filename}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
