import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import datetime
import copy
import pandas as pd
import numpy as np
import sys        
import pygame


# Import configuration modules
from config_gui import (DEFAULT_CONFIG, get_user_parameters, generate_random_parameters)

# List of algorithms to test
import algo_pomcp
import algo_pomcp_shr  
import algo_cen_pomcp
import algo_auction
import algo_greedy

ALGORITHMS = [
    ("DEC-POMCP", algo_pomcp),
    ("SHR-POMCP", algo_pomcp_shr),
    ("CEN-POMCP", algo_cen_pomcp),
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

    # 1. MANUAL OR AUTOMATIC SCENARIO CONFIGURATION
    all_scenarios_params = []
    
    for i in range(1, num_scenarios + 1):
        if scelta_config == '1':
            print(f"\n{'='*50}")
            print(f" STARTING SCENARIO CONFIGURATION {i}/{num_scenarios}")
            print(f"{'='*50}")
            params = get_user_parameters(scenario_idx=i)
            params['use_gui'] = True
        else:
            params = generate_random_parameters(scenario_idx=i)
            params['use_gui'] = False
        all_scenarios_params.append(params)
        
    
    # 2. ITERATIONS OVER ALPHA VALUES, SCENARIOS AND ALGORITHMS
    try:

        for alpha_idx, current_alpha in enumerate(alpha_values):
            print(f"\n\n{'#'*60}")
            print(f" STARTING SIMULATIONS WITH REWARD_ALPHA = {current_alpha} ")
            print(f"{'#'*60}")

            results_data = []

            for sc_idx, params in enumerate(all_scenarios_params):
                scenario_idx = params['scenario_idx']
            
                remaining_scenarios_current_alpha = len(all_scenarios_params) - sc_idx
                remaining_scenarios_future_alphas = (len(alpha_values) - alpha_idx - 1) * len(all_scenarios_params)
                total_remaining_scenarios = remaining_scenarios_current_alpha + remaining_scenarios_future_alphas
            
                max_t = params.get('max_time', DEFAULT_CONFIG.get('max_time', 2.5))
                num_pomcp_algorithms = 3
                avg_iterations = 50
            
                remaining_seconds = avg_iterations * num_pomcp_algorithms * max_t * total_remaining_scenarios
                estimated_time = str(datetime.timedelta(seconds=int(remaining_seconds)))
            
                print(f" Estimated remaining time: ~ {estimated_time} ")
                
                
                scenario_record = {
                    "Scenario": scenario_idx,
                    "Drone Pos": str(params.get('drone_positions', [])),
                    "Target Pos": str(params.get('target_pos', '')),
                    "Num Peaks": len(params.get('peaks', [])),
                    "Peak Means": str([p['mean'] for p in params.get('peaks', [])]),
                    "Peak Vars": str([p['cov'] for p in params.get('peaks', [])]),
                    "Obstacles": str(params.get('obstacles', [])),
                    "Step POMCP": None,
                    "Step SHR-POMCP": None,
                    "Step CEN-POMCP": None,
                    "Step AUCTION": None,
                    "Step GREEDY": None,
                    "pomcp_metrics": [],
                    "shr_pomcp_metrics": [],
                    "cen_pomcp_metrics": []
                }

                for algo_name, algo_module in ALGORITHMS:
                    print(f"\n Executing: {algo_name} (Scenario {scenario_idx} | Alpha {current_alpha})")
                    
                    current_params = copy.deepcopy(params)
                    current_params['algo_name'] = algo_name
                    current_params['save_folder'] = session_folder
                    current_params['reward_alpha'] = current_alpha
                    
                    res = algo_module.run_simulation(current_params)
                    steps_taken = res[0]
                    
                    if steps_taken == -1:
                        print(f"Simulation {algo_name} stopped by user.")
                        steps_taken = float('inf') 
                    
                    if algo_name == "DEC-POMCP":
                        scenario_record["Step POMCP"] = steps_taken
                        scenario_record["pomcp_metrics"] = res[2] if len(res) > 2 else []
                    elif algo_name == "SHR-POMCP":
                        scenario_record["Step SHR-POMCP"] = steps_taken
                        scenario_record["shr_pomcp_metrics"] = res[2] if len(res) > 2 else []
                    elif algo_name == "CEN-POMCP":
                        scenario_record["Step CEN-POMCP"] = steps_taken
                        scenario_record["cen_pomcp_metrics"] = res[2] if len(res) > 2 else []
                    elif algo_name == "AUCTION":
                        scenario_record["Step AUCTION"] = steps_taken
                    elif algo_name == "GREEDY":
                        scenario_record["Step GREEDY"] = steps_taken

                results_data.append(scenario_record)

            # Generate complete excel report
            generate_excel_report(results_data, session_folder, global_timestamp, current_alpha)

    except KeyboardInterrupt:
        
        pygame.quit() 
        print(" Interruption of simulations by user. ")
        
        # Save partial data on excel 
        if 'results_data' in locals() and results_data:
            print("Partial report generation...")
            try:
                alpha_str = current_alpha if 'current_alpha' in locals() else "Sconosciuto"
                generate_excel_report(results_data, session_folder, global_timestamp, f"{alpha_str}_INTERROTTO")
            except Exception as e:
                print(f"Impossible to save partial report: {e}")
        
        sys.exit(0)


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

    # SHEET 2: ALGORITHM PERFORMANCE COMPARISON
    performance_records = []
    win_counts = {"DEC-POMCP": 0, "SHR-POMCP": 0, "CEN-POMCP": 0, "AUCTION": 0, "GREEDY": 0}
    
    for row in data:
        steps_dict = {
            "DEC-POMCP": row["Step POMCP"],
            "SHR-POMCP": row["Step SHR-POMCP"],
            "CEN-POMCP": row["Step CEN-POMCP"],
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
            winner_str = "No winner (all interrupted or failed)"

        rec = {"Scenario": row["Scenario"]}
        rec.update({k: (v if v != float('inf') else "Interruption") for k,v in steps_dict.items()})
        rec["Winners"] = winner_str
        performance_records.append(rec)

    total_row = {"Scenario": "Total Wins"}
    total_row.update(win_counts)
    total_row["Winners"] = ""
    performance_records.append(total_row)

    df_perf = pd.DataFrame(performance_records)
    df_perf.to_excel(writer, sheet_name='Performance', index=False)
    worksheet_perf = writer.sheets['Performance']
    for i, col in enumerate(df_perf.columns):
        worksheet_perf.set_column(i, i, max(len(col)+2, 15))


    # Head-to-Head Charts

    # Dictionary to map algorithm names to dictionary keys
    algo_keys = {
        "DEC-POMCP": "Step POMCP",
        "SHR-POMCP": "Step SHR-POMCP",
        "AUCTION": "Step AUCTION",
        "GREEDY": "Step GREEDY"
    }

    # The 5 comparisons
    comparisons = [
        ("DEC-POMCP", "GREEDY"),
        ("DEC-POMCP", "AUCTION"),
        ("DEC-POMCP", "SHR-POMCP"),
        ("SHR-POMCP", "AUCTION"),
        ("SHR-POMCP", "GREEDY")
    ]

    data_col = 20 # Column U (hidden far right to store the chart data)
    data_row = 1
    
    chart_row_1 = len(performance_records) + 3   # Row 1 of charts
    chart_row_2 = chart_row_1 + 16               # Row 2 of charts (below row 1)

    # Grid positions for the 5 charts (row, col)
    # Top row: 3 charts spaced out
    # Bottom row: 2 charts centered
    positions = [
        (chart_row_1, 0),   # Col A
        (chart_row_1, 5),   # Col F
        (chart_row_1, 10),  # Col K
        (chart_row_2, 2),   # Col C (centered)
        (chart_row_2, 7)    # Col H (centered)
    ]

    for idx, (algo1, algo2) in enumerate(comparisons):
        w1, w2, draws = 0, 0, 0
        
        # Calculate Head-to-Head for all scenarios
        for row in data:
            v1 = row.get(algo_keys[algo1])
            v2 = row.get(algo_keys[algo2])
            
            # Treat None as infinity (failed/interrupted)
            v1_val = v1 if v1 is not None else float('inf')
            v2_val = v2 if v2 is not None else float('inf')
            
            if v1_val < v2_val:
                w1 += 1
            elif v2_val < v1_val:
                w2 += 1
            else:
                if v1_val != float('inf'): # Count draw only if both finished
                    draws += 1

        # Write data to the hidden table area
        worksheet_perf.write(data_row, data_col, algo1)
        worksheet_perf.write(data_row+1, data_col, w1)

        worksheet_perf.write(data_row, data_col+1, algo2)
        worksheet_perf.write(data_row+1, data_col+1, w2)

        worksheet_perf.write(data_row, data_col+2, "Draws")
        worksheet_perf.write(data_row+1, data_col+2, draws)

        # Create the column chart
        chart = workbook.add_chart({'type': 'column'})
        
        # Add series and color the bars (Blue for Algo1, Red for Algo2, Grey for Draws)
        chart.add_series({
            'categories': ['Performance', data_row, data_col, data_row, data_col+2],
            'values':     ['Performance', data_row+1, data_col, data_row+1, data_col+2],
            'data_labels': {'value': True},
            'points': [
                {'fill': {'color': '#4F81BD'}}, 
                {'fill': {'color': '#C0504D'}}, 
                {'fill': {'color': '#A6A6A6'}}  
            ]
        })
        
        chart.set_title({'name': f"{algo1} vs {algo2}"})
        chart.set_y_axis({'name': 'Head-to-Head Wins'})
        chart.set_legend({'none': True})
        chart.set_size({'width': 350, 'height': 280}) # Keep it compact to fit side-by-side

        # Insert chart in the designated position
        ws_row, ws_col = positions[idx]
        worksheet_perf.insert_chart(ws_row, ws_col, chart)

        data_row += 3 # Move down for the next chart's data

    # =========================================================================
    # SHEET 3: POMCP METRICS FOR ALL VERSIONS
    # =========================================================================
    metrics_records = []
    for row in data:
        scenario_num = row["Scenario"]
        
        # Metrics extraction from algorithms 
        for algo_label, metric_key in [("DEC-POMCP", "pomcp_metrics"), ("SHR-POMCP", "shr_pomcp_metrics"), ("CEN-POMCP", "cen_pomcp_metrics")]:
            all_metrics = row[metric_key]
            
            m_list = []
            
            if all_metrics:
                if algo_label in ["DEC-POMCP", "SHR-POMCP"] and 1 in all_metrics:
                    m_list = [m for m in all_metrics[1] if m is not None]
                elif algo_label == "CEN-POMCP":
                    m_list = [m for m in all_metrics if m is not None]
            
            if m_list:
                iters = [m['iterations'] for m in m_list]
                depths = [m['depth'] for m in m_list]
                nodes = [m['nodes'] for m in m_list]
                
                metrics_records.append({
                    "Scenario": f"{scenario_num} ({algo_label})",
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
                metrics_records.append({"Scenario": f"{scenario_num} ({algo_label})"})
            
    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_excel(writer, sheet_name='Metrics_Tree', index=False)
    worksheet_met = writer.sheets['Metrics_Tree']
    for i, col in enumerate(df_metrics.columns):
        worksheet_met.set_column(i, i, max(len(col)+2, 12))

    # =========================================================================
    # Helper functions for charts 
    # =========================================================================
    def create_spaced_chart_sheet_decentralized(sheet_name, data, metric_key, chart_title, y_axis_name, data_source_key):
        ws = workbook.add_worksheet(sheet_name)
        row_offset = 0
        
        for row in data:
            scenario = row["Scenario"]
            metrics_dict = row[data_source_key]
            if not metrics_dict: continue
                
            max_steps = max((len(m_list) for m_list in metrics_dict.values()), default=0)
            if max_steps == 0: continue
                
            ws.write(row_offset, 0, f"Scenario {scenario}")
            for step_idx in range(max_steps):
                ws.write(row_offset, step_idx + 1, f"Step {step_idx+1}")
            
            chart = workbook.add_chart({'type': 'line'})
            current_row = row_offset + 1
            
            for drone_id, m_list in metrics_dict.items():
                ws.write(current_row, 0, f"Drone {drone_id}")
                vals = [m.get(metric_key) if m is not None else None for m in m_list]
                for col_idx, val in enumerate(vals):
                    if val is not None:
                        ws.write(current_row, col_idx + 1, round(val, 2))
                
                chart.add_series({
                    'name': f'Drone {drone_id}',
                    'categories': [sheet_name, row_offset, 1, row_offset, max_steps],
                    'values': [sheet_name, current_row, 1, current_row, max_steps],
                    'marker': {'type': 'circle', 'size': 4},
                    'line': {'width': 1.5}
                })
                current_row += 1
            
            chart.set_title({'name': f'{chart_title} - Scenario {scenario}'})
            chart.set_x_axis({'name': 'Time Steps (Global)'})
            chart.set_y_axis({'name': y_axis_name})
            chart.set_size({'width': 700, 'height': 350})
            ws.insert_chart(current_row + 1, 1, chart)
            row_offset = current_row + 22

    def create_spaced_chart_sheet_centralized(sheet_name, data, metric_key, chart_title, y_axis_name, data_source_key):
        ws = workbook.add_worksheet(sheet_name)
        row_offset = 0
        
        for row in data:
            scenario = row["Scenario"]
            metrics_list = row[data_source_key] 
            if not metrics_list: continue
                
            max_steps = len(metrics_list)
            if max_steps == 0: continue
                
            ws.write(row_offset, 0, f"Scenario {scenario}")
            for step_idx in range(max_steps):
                ws.write(row_offset, step_idx + 1, f"Step {step_idx+1}")
            
            chart = workbook.add_chart({'type': 'line'})
            current_row = row_offset + 1
            
            ws.write(current_row, 0, "Team Centrale")
            vals = [m.get(metric_key) if m is not None else None for m in metrics_list]
            for col_idx, val in enumerate(vals):
                if val is not None:
                    ws.write(current_row, col_idx + 1, round(val, 2))
            
            chart.add_series({
                'name': 'Team Centrale',
                'categories': [sheet_name, row_offset, 1, row_offset, max_steps],
                'values': [sheet_name, current_row, 1, current_row, max_steps],
                'marker': {'type': 'circle', 'size': 4},
                'line': {'width': 2.0}
            })
            current_row += 1
            
            chart.set_title({'name': f'{chart_title} - Scenario {scenario}'})
            chart.set_x_axis({'name': 'Time Steps (Global)'})
            chart.set_y_axis({'name': y_axis_name})
            chart.set_size({'width': 700, 'height': 350})
            ws.insert_chart(current_row + 1, 1, chart)
            row_offset = current_row + 22

    # SHEET 4 & 5: DEC-POMCP (EXPLORATION RATIO AND ROOT FLIPS)
    create_spaced_chart_sheet_decentralized('Expl_Ratio', data, 'expl_ratio', 'Exploration vs Exploitation (DEC-POMCP)', 'Ratio %', 'pomcp_metrics')
    create_spaced_chart_sheet_decentralized('Root_Flips', data, 'flips', 'Variazioni Azione Root (DEC-POMCP)', 'Numero Flips', 'pomcp_metrics')

    # SHEET 6 & 7: SHR-POMCP (EXPLORATION RATIO AND ROOT FLIPS)
    create_spaced_chart_sheet_decentralized('Expl_Ratio_SHR', data, 'expl_ratio', 'Exploration vs Exploitation (SHR-POMCP)', 'Ratio %', 'shr_pomcp_metrics')
    create_spaced_chart_sheet_decentralized('Root_Flips_SHR', data, 'flips', 'Variazioni Azione Root (SHR-POMCP)', 'Numero Flips', 'shr_pomcp_metrics')

    # SHEET 8 & 9: CEN-POMCP (EXPLORATION RATIO AND ROOT FLIPS)
    create_spaced_chart_sheet_centralized('Expl_Ratio_CEN', data, 'expl_ratio', 'Exploration vs Exploitation (CEN-POMCP)', 'Ratio %', 'cen_pomcp_metrics')
    create_spaced_chart_sheet_centralized('Root_Flips_CEN', data, 'flips', 'Variazioni Azione Root (CEN-POMCP)', 'Numero Flips', 'cen_pomcp_metrics')

    writer.close()
    print(f"\n{'='*50}")
    print(f"Simulation completed! Report Excel saved as:\n{filename}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()