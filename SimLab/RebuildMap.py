import os
import sys
import ast
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import pygame
from scipy.stats import multivariate_normal

# =============================================================================
# GLOBAL CONSTANTS & COLORS
# =============================================================================
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GRAY': (200, 200, 200),
    'LIGHT_GRAY': (240, 240, 240),
    'BLUE': (50, 100, 255),
    'RED': (255, 50, 50),
    'PURPLE': (200, 50, 255),
    'ORANGE': (255, 140, 0)
}

# =============================================================================
# BELIEF MAP GENERATION (Reused from simulator)
# =============================================================================
def initialize_obstacle_map(params):
    """Creates a 2D numpy array representing obstacles on the grid."""
    map_size = params['map_size']
    obstacle_map = np.zeros((map_size, map_size), dtype=int)
    for obs_pos in params.get('obstacles', []):
        r, c = obs_pos
        obstacle_map[r, c] = 1
    return obstacle_map

def initialize_belief_map(params):
    """Creates the initial probability distribution based on Gaussian peaks."""
    map_size = params['map_size']
    peaks = params.get('peaks', [])
    
    belief_map = np.zeros((map_size, map_size))
    
    # If no peaks are present, it's a uniform map
    if not peaks:
        belief_map.fill(1.0)
    else:
        x, y = np.mgrid[0:map_size, 0:map_size]
        coord = np.dstack((x, y))
        for peak in peaks:
            mean = peak['mean']
            sigmas = peak['cov']
            cov_matrix = [[sigmas[0]**2, 0], [0, sigmas[1]**2]]
            rv = multivariate_normal(mean, cov_matrix)
            belief_map += rv.pdf(coord)

    # Zero out probability over obstacles
    if 'obstacles' in params and len(params['obstacles']) > 0:
        obstacle_map = initialize_obstacle_map(params)
        belief_map = belief_map * (1 - obstacle_map)
    
    # Normalize map
    sum_prob = np.sum(belief_map)
    if sum_prob == 0:
        belief_map.fill(1.0 / (map_size * map_size))
    else:
        belief_map /= sum_prob
        
    return belief_map

# =============================================================================
# PYGAME RENDERING LOGIC
# =============================================================================
def render_scenario(params, excel_dir, scenario_idx):
    """Initializes Pygame and renders the parsed scenario configuration."""
    pygame.init()
    
    map_size = params['map_size']
    display_info = pygame.display.Info()
    
    # Calculate cell size dynamically to fit the screen
    cell_size = min(int(display_info.current_w * 0.80) // map_size, 
                    int(display_info.current_h * 0.80) // map_size)
    
    screen_w = map_size * cell_size
    screen_h = map_size * cell_size
    
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"Reconstructed Map - Scenario {scenario_idx} | ESC to Exit | R for Screenshot")
    
    font_cell = pygame.font.SysFont(None, max(1, cell_size // 3))
    clock = pygame.time.Clock()
    
    # Pre-compute belief and obstacle maps
    belief_map = initialize_belief_map(params)
    obstacle_map = initialize_obstacle_map(params)
    max_prob = belief_map.max()
    
    drone_colors = list(COLORS.values())[8:] if len(COLORS) > 8 else [COLORS['BLUE'], COLORS['ORANGE'], COLORS['PURPLE'], COLORS['GRAY']]
    
    running = True
    while running:
        screen.fill(COLORS['WHITE'])
        
        # 1. Draw Background (Probability Heatmap & Obstacles)
        for r in range(map_size):
            for c in range(map_size):
                x = c * cell_size
                y = r * cell_size
                prob = belief_map[r, c]

                if obstacle_map[r, c] == 1:
                    color = COLORS['BLACK']
                else:
                    if max_prob > 1e-9:
                        # Blue shading based on probability concentration
                        color_val = int(255 * ((prob / max_prob) ** 0.4))
                        color = (255 - color_val, 255 - color_val, 255)
                    else:
                        color = COLORS['WHITE']

                pygame.draw.rect(screen, color, (x, y, cell_size, cell_size))
                pygame.draw.rect(screen, COLORS['BLACK'], (x, y, cell_size, cell_size), 1)

                # Draw probability text
                if obstacle_map[r, c] == 0:
                    text = font_cell.render(f"{prob * 100:.2f}%", True, COLORS['BLACK'])
                    text_rect = text.get_rect(centerx=x + cell_size // 2, bottom=y + cell_size - max(2, cell_size // 20))
                    screen.blit(text, text_rect)

        # 2. Draw Gaussian Peaks (Optional: Mark with 'P')
        for peak in params.get('peaks', []):
            pr, pc = peak['mean']
            center = (pc * cell_size + cell_size // 2, pr * cell_size + cell_size // 2)
            peak_label = font_cell.render("P", True, COLORS['PURPLE'])
            screen.blit(peak_label, peak_label.get_rect(center=(center[0], center[1] - cell_size//4)))

        # 3. Draw Target
        if params.get('target_pos'):
            tr, tc = params['target_pos']
            target_rect = pygame.Rect(tc * cell_size, tr * cell_size, cell_size, cell_size)
            pygame.draw.line(screen, COLORS['RED'], target_rect.topleft, target_rect.bottomright, 3)
            pygame.draw.line(screen, COLORS['RED'], target_rect.topright, target_rect.bottomleft, 3)

        # 4. Draw Drones
        for idx, pos in enumerate(params.get('drone_positions', [])):
            dr, dc = pos
            center = (dc * cell_size + cell_size // 2, dr * cell_size + cell_size // 2)
            color = drone_colors[idx % len(drone_colors)]
            pygame.draw.circle(screen, color, center, cell_size // 3, 4)
            
            id_text = font_cell.render(str(idx + 1), True, color)
            screen.blit(id_text, id_text.get_rect(center=center))

        pygame.display.flip()
        
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Exit the current visualization
                    running = False
                    
                if event.key == pygame.K_r:
                    # Save Screenshot
                    screenshot_name = f"Screen scenario {scenario_idx}.png"
                    save_path = os.path.join(excel_dir, screenshot_name)
                    pygame.image.save(screen, save_path)
                    print(f"\n[✓] Screenshot saved successfully at:\n{save_path}")

    pygame.quit()

# =============================================================================
# MAIN LOGIC
# =============================================================================
def main():
    # Hide the main tkinter root window
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the Excel file generated by the simulator")
    file_path = filedialog.askopenfilename(
        title="Select Simulation Excel File",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    
    if not file_path:
        print("No file selected. Exiting...")
        return

    excel_dir = os.path.dirname(file_path)
    print(f"Loaded file: {file_path}")

    try:
        # Read the 'Parametri' sheet
        df = pd.read_excel(file_path, sheet_name='Parametri')
    except Exception as e:
        print(f"Error loading Excel file, ensure it contains the 'Parametri' sheet. Error: {e}")
        return

    while True:
        try:
            print("\n" + "="*50)
            map_size_input = input("Enter Map Size: ")
            map_size = int(map_size_input)
            
            scenario_input = input("Which Scenario ID would you like to reconstruct?: ")
            scenario_idx = int(scenario_input)
            
            # Find the row corresponding to the scenario
            scenario_row = df[df['Scenario'] == scenario_idx]
            
            if scenario_row.empty:
                print(f"Scenario {scenario_idx} not found in the Excel file. Please try again.")
                continue
                
            row = scenario_row.iloc[0]
            
            # Extract and parse columns using ast.literal_eval for safe string evaluation
            drone_pos = ast.literal_eval(row['Drone Pos'])
            target_pos = ast.literal_eval(row['Target Pos'])
            peak_means = ast.literal_eval(row['Peak Means'])
            peak_vars = ast.literal_eval(row['Peak Vars'])
            obstacles = ast.literal_eval(row['Obstacles'])
            
            # Reconstruct peaks dictionary structure
            peaks = [{'mean': mean, 'cov': cov} for mean, cov in zip(peak_means, peak_vars)]
            
            # Build parameter dictionary for rendering
            params = {
                'map_size': map_size,
                'drone_positions': drone_pos,
                'target_pos': target_pos,
                'peaks': peaks,
                'obstacles': obstacles
            }
            
            print(f"\n[+] Reconstructing Scenario {scenario_idx}...")
            print("  - Press 'ESC' in the window to close it and load a new scenario.")
            print("  - Press 'R' to save a screenshot.")
            
            # Start Pygame rendering
            render_scenario(params, excel_dir, scenario_idx)

        except ValueError:
            print("Invalid input. Please enter numerical values for map size and scenario ID.")
        except KeyboardInterrupt:
            # Clean exit on CTRL+C
            print("\nExiting map reconstructor...")
            break
        except Exception as e:
            print(f"An unexpected error occurred parsing the data: {e}")

if __name__ == "__main__":
    main()