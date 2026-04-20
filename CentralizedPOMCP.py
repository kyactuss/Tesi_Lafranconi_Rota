import elkai
from collections import deque
import math
from scipy.ndimage import label, distance_transform_edt
import random
import time
import numpy as np
from scipy.stats import multivariate_normal
import pygame
import itertools
import sys

# =============================================================================
# 1. PARAMETERS CONFIGURATION 
# =============================================================================
DEFAULT_CONFIG = {
    'map_size': 20,
    'alpha_sensor': 0.01,
    'beta_sensor': 0.01,
    'max_time': 5, # Ridotto per simulazione centralizzata fluida
    'depth_limit': 15, # Profondità ridotta causa esplosione combinatoria azioni
    'discount_factor': 0.98,
    'exploration_const': math.sqrt(2),
    'reward_alpha': 3,
    'explorative_reward': 0.005,
    'r_target': 1,
}

MOVES_DELTA = {
    'N': (-1, 0),
    'S': (1, 0),
    'W': (0, -1),
    'E': (0, 1),
    'Stay': (0, 0)
}

COLORS = {
    'WHITE': (255, 255, 255), 'BLACK': (0, 0, 0), 'GRAY': (200, 200, 200),
    'LIGHT_GRAY': (240, 240, 240), 'BLUE': (50, 100, 255), 'RED': (255, 50, 50),
    'GREEN': (0, 200, 0), 'PURPLE': (200, 50, 255), 'ORANGE': (255, 140, 0),
    'LIGHT_BLUE': (0, 0, 255), 'LIGHT_RED': (200, 0, 0)
}

DARP_AREA_COLORS = [
    (144, 238, 144), (255, 160, 160), (216, 191, 216), (173, 216, 230),
    (255, 255, 204), (210, 180, 140), (211, 211, 211), (255, 218, 185)
]

# Funzione per raccolta parametri utente tramite interfaccia grafica
def get_user_parameters():
    
    print("=== SEARCH MISSION CONFIGURATION ===")
    
    map_size = DEFAULT_CONFIG['map_size']
    alpha_sensor = DEFAULT_CONFIG['alpha_sensor']
    beta_sensor = DEFAULT_CONFIG['beta_sensor']

    print(f"Parameters: Map Size={map_size}x{map_size}, Alpha={alpha_sensor}, Beta={beta_sensor}")
    
    # Initialize pygame for configuration graphical interface
    pygame.init()
    
    # Get screen dimensions to adapt the window
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    
    # Fixed dimensions
    margin = 25  
    info_height = 55  
    bottom_margin = 35
    
    # Calculate maximum usable space
    max_usable_width = screen_width * 0.9
    max_usable_height = screen_height * 0.9
    
    # Calculate cell size
    cell_size_from_width = int((max_usable_width - 2 * margin) / map_size)
    cell_size_from_height = int((max_usable_height - info_height - margin - bottom_margin) / map_size)
    cell_size = min(cell_size_from_width, cell_size_from_height)
    
    window_width = map_size * cell_size + 2 * margin
    window_height = map_size * cell_size + info_height + margin + bottom_margin
    
    # Create the window
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Mission Configuration - Click to select")
    
    # Fonts
    font_title = pygame.font.Font(None, 28)
    font_info = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 18)
    font_selected = pygame.font.Font(None, 20)
    font_cell = pygame.font.Font(None, max(1, int(cell_size*0.5)))  # Dynamic font for cell coordinates
    
    # Function to draw the grid with selected positions and elements selected in previous phases
    def draw_grid(selected_positions, phase_name, phase_color, context_items=None):
        
        screen.fill(COLORS['WHITE'])
        
        if context_items is None:
            context_items = {}
        
        # Extract only the positions of the peaks, not the full dict, for easier checking in grid drawing
        if 'peaks' in context_items and context_items['peaks']:
            if isinstance(context_items['peaks'][0], dict):
                peaks_positions = [peak['mean'] for peak in context_items['peaks']]
            else:
                peaks_positions = context_items['peaks']
        else:
            peaks_positions = []
        
        # Extract only the spatial coordinates of traces
        if 'traces' in context_items and context_items['traces']:
            traces_positions = [trace['pos'] for trace in context_items['traces']]
        else:
            traces_positions = []
        
        # Title and instructions
        title_text = font_title.render(f"PHASE: {phase_name}", True, phase_color)
        info_text = font_info.render("Click on grid to select cells", True, COLORS['BLACK'])
        help_text = font_small.render("Press ENTER to confirm and proceed to next phase", True, COLORS['RED'])

        screen.blit(title_text, (margin, 10))
        screen.blit(info_text, (margin, 35))
        screen.blit(help_text, (margin, 60))
        
        # Grid drawing
        grid_offset_y = margin + info_height
        for r in range(map_size):
            for c in range(map_size):
                x = margin + c * cell_size
                y = grid_offset_y + r * cell_size
                
                # Determine cell color and label
                cell_color = COLORS['LIGHT_GRAY']
                label = f"{r},{c}"
                text_color = COLORS['BLACK']
                
                # Check if it's an already placed drone
                if 'drones' in context_items and (r, c) in context_items['drones']:
                    cell_color = COLORS['BLUE']
                    label = "D"
                    text_color = COLORS['WHITE']
                # Check if it's the already placed target
                elif 'target' in context_items and (r, c) == context_items['target']:
                    cell_color = COLORS['RED']
                    label = "T"
                    text_color = COLORS['WHITE']
                # Check if it's an already placed Gaussian peak
                elif (r, c) in peaks_positions:
                    cell_color = COLORS['PURPLE']
                    label = "G"
                    text_color = COLORS['WHITE']
                # Check if it's an already placed trace
                elif (r, c) in traces_positions:
                    cell_color = COLORS['ORANGE']
                    label = "Tr"
                    text_color = COLORS['WHITE']
                # Check if it's an already placed obstacle
                elif 'obstacles' in context_items and (r, c) in context_items['obstacles']:
                    cell_color = COLORS['BLACK']
                    label = "X"
                    text_color = COLORS['WHITE']
                # Check if the cell is selected in the current phase
                elif (r, c) in selected_positions:
                    cell_color = phase_color
                    label = f"{r},{c}"
                    text_color = COLORS['WHITE']
                
                # Draw the cell
                pygame.draw.rect(screen, cell_color, (x, y, cell_size, cell_size))
                pygame.draw.rect(screen, COLORS['GRAY'], (x, y, cell_size, cell_size), 1)
                
                # Show coordinates or label at the center of the cell
                coord_text = font_cell.render(label, True, text_color)
                text_rect = coord_text.get_rect(center=(x + cell_size//2, y + cell_size//2))
                screen.blit(coord_text, text_rect)
        
        # Selected cells counter
        count_text = font_selected.render(f"Selected: {len(selected_positions)}", True, COLORS['BLACK'])
        screen.blit(count_text, (margin, window_height - bottom_margin + 10))
        
        pygame.display.flip()
    
    # Function to convert mouse position to grid coordinates (row, column)
    def get_cell_from_mouse(mouse_pos):
        
        grid_offset_y = margin + info_height
        x, y = mouse_pos
        
        if x < margin or y < grid_offset_y:
            return None
        
        c = (x - margin) // cell_size
        r = (y - grid_offset_y) // cell_size
        
        if 0 <= r < map_size and 0 <= c < map_size:
            return (r, c)
        return None
    
    # Function to select cells interactively for each phase, handles selection loop and visualization
    def interactive_selection(phase_name, phase_color, allow_multiple=True, context_items=None, validate_cell=False, allow_empty=False):
        
        selected = []
        running = True
        
        while running:
            draw_grid(selected, phase_name, phase_color, context_items)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    cell = get_cell_from_mouse(event.pos)
                    if cell:
                        if cell in selected:
                            selected.remove(cell)  # Deselect if already selected
                        else:
                            # Validation for obstacles: avoid the choice of cells with drones or target
                            if validate_cell:
                                if cell in context_items.get('drones', []) or cell == context_items.get('target'):
                                    print(f"  Cannot place obstacle on drone/target!")
                                    continue
                            
                            if allow_multiple:
                                selected.append(cell) # Add to selection of multiple cells
                            else:
                                selected = [cell]  # Only one selection is allowed
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if len(selected) > 0 or allow_empty:
                            running = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
        
        return selected
    
    # dictionary to store all selected parameters and positions during the configuration phases
    dictionary_context={}

    # PHASE 1: Drone positions selection
    print("\n[1/6] Select DRONES positions ")
    dictionary_context['drones'] = interactive_selection("1. DRONES Positions", COLORS['BLUE'], allow_multiple=True)
    print(f"✓ {len(dictionary_context['drones'])} drones configured")
    
    # PHASE 2: Target position selection
    print("\n[2/6] Select TARGET position ")
    dictionary_context['target'] = interactive_selection("2. TARGET Position", COLORS['RED'], allow_multiple=False, context_items=dictionary_context)[0]
    print(f"✓ Target at position {dictionary_context['target']}")
    
    # PHASE 3: Initial belief map type selection
    print("\n[3/6] Choose initial belief map type")

    screen.fill(COLORS['WHITE'])
    title = font_title.render("Initial Belief Map Type", True, COLORS['BLACK'])
    screen.blit(title, (margin, margin))
   
    options = [
        "1 - Uniform (press 1)",
        "2 - Multi-Gaussian (press 2) - From 1 to multiple peaks"
    ]
    
    for i, opt in enumerate(options):
        text = font_info.render(opt, True, COLORS['BLACK'])
        screen.blit(text, (margin, 80 + i * 40))  # Fixed vertical positions
    
    pygame.display.flip()
    
    dictionary_context['map_type'] = None
    waiting = True

    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    dictionary_context['map_type'] = 1
                    waiting = False
                elif event.key == pygame.K_2:
                    dictionary_context['map_type'] = 2
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
    
    dictionary_context['peaks'] = []

    print(f"✓ Map type: {['Uniform', 'Multi-Gaussian'][dictionary_context['map_type']-1]}")
    
    # PHASE 4: Probability peaks selection (if not uniform)
    if dictionary_context['map_type'] == 2:
        print("\n[4/6] Select GAUSSIAN centers (minimum 1, multiple clicks + ENTER)")
        
        # For each peak, ask for sigma (via terminal)
        for i, center in enumerate(interactive_selection("4. GAUSSIAN Centers", COLORS['PURPLE'], allow_multiple=True, context_items=dictionary_context)):
            print(f"\n  Peak #{i+1} at position {center}")
            while True:
                try:
                    sigmas = input("  Enter Sigma_X, Sigma_Y (e.g., 2.0,2.0): ")
                    sx, sy = map(float, sigmas.split(','))
                    if sx > 0 and sy > 0:
                        break
                    print("  Standard deviations must be positive.")
                except ValueError:
                    print("  Invalid format. Use: number,number")
            
            dictionary_context['peaks'].append({
                'mean': center,
                'cov': [sx, sy]
            })
        
        print(f"✓ {len(dictionary_context['peaks'])} Gaussian peaks configured")
    else:
        print("\n[4/6] No Gaussian peak needed (uniform map)")
    
    # PHASE 5: Traces position selection
    print("\n[5/6] Select TRACES positions (multiple clicks + ENTER, or just ENTER for none)")
    selected_traces_coords = interactive_selection("5. TRACES Positions", COLORS['ORANGE'], allow_multiple=True, context_items=dictionary_context, validate_cell=True, allow_empty=True)
    
    dictionary_context['traces'] = []
    
    # For each selected trace position, ask for type and parameters
    for i, coord in enumerate(selected_traces_coords):
        print(f"\n  Trace #{i+1} at position {coord}")
        print("  Select trace type:")
        print("    1 - Von Mises")
        print("    2 - Ring")
        print("    3 - Gaussian")
        
        while True:
            try:
                trace_type_input = input("  Enter choice (1/2/3): ").strip()
                if trace_type_input in ['1', '2', '3']:
                    break
                print("  Invalid choice. Please enter 1, 2, or 3.")
            except ValueError:
                print("  Invalid input.")
        
        # Collect parameters based on trace type
        if trace_type_input == '1':  # Von Mises
            while True:
                try:
                    mu_deg = float(input("  Enter direction μ (in degrees): "))
                    kappa = float(input("  Enter concentration κ: "))
                    if kappa >= 0:
                        break
                    print("  Concentration must be non-negative.")
                except ValueError:
                    print("  Invalid format. Please enter valid numbers.")

            mu = math.radians(mu_deg) # Convert direction to radians for internal use
            
            trace_dict = {
                'pos': coord,
                'type': 'von_mises',
                'trace_params': {'mu': mu, 'kappa': kappa}
            }
        
        elif trace_type_input == '2':  # Ring
            while True:
                try:
                    radius = float(input("  Enter radius: "))
                    variance = float(input("  Enter variance: "))
                    if radius > 0 and variance > 0:
                        break
                    print("  Radius and variance must be positive.")
                except ValueError:
                    print("  Invalid format. Please enter valid numbers.")
            
            trace_dict = {
                'pos': coord,
                'type': 'ring',
                'trace_params': {'radius': radius, 'variance': variance}
            }
        
        elif trace_type_input == '3':  # Gaussian
            while True:
                try:
                    sigmas = input("  Enter Sigma_X, Sigma_Y (e.g., 2.0,2.0): ")
                    sigma_x, sigma_y = map(float, sigmas.split(','))
                    if sigma_x > 0 and sigma_y > 0:
                        break
                    print("  Standard deviations must be positive.")
                except ValueError:
                    print("  Invalid format. Use: number,number")
            
            trace_dict = {
                'pos': coord,
                'type': 'gaussian',
                'trace_params': {'sigma_x': sigma_x, 'sigma_y': sigma_y}
            }
        
        dictionary_context['traces'].append(trace_dict)
    
    print(f"✓ {len(dictionary_context['traces'])} traces configured")
    
    # PHASE 6: Obstacles position selection
    print("\n[6/6] Select OBSTACLES positions (multiple clicks + ENTER, or just ENTER for none)")   
    dictionary_context['obstacles'] = interactive_selection("6. OBSTACLES Positions", COLORS['BLACK'], allow_multiple=True, context_items=dictionary_context, validate_cell=True, allow_empty=True)
    print(f"✓ {len(dictionary_context['obstacles'])} obstacles configured")

    # Close pygame window
    pygame.quit()
    
    print("\n=== CONFIGURATION COMPLETED ===")
    print(f"Drones: {len(dictionary_context['drones'])}")
    print(f"Target: {dictionary_context['target']}")
    print(f"Map type: {['Uniform', 'Multi-Gaussian'][dictionary_context['map_type']-1]}")
    print(f"Obstacles: {len(dictionary_context['obstacles'])}")

    # Compile all parameters into a single dictionary
    params_dict = {
        'map_size': map_size,
        'alpha_sensor': alpha_sensor,
        'beta_sensor': beta_sensor,
        'max_time': DEFAULT_CONFIG['max_time'],
        'depth_limit': DEFAULT_CONFIG['depth_limit'],
        'discount_factor': DEFAULT_CONFIG['discount_factor'],
        'exploration_const': DEFAULT_CONFIG['exploration_const'],
        'reward_alpha': DEFAULT_CONFIG['reward_alpha'],
        'explorative_reward': DEFAULT_CONFIG['explorative_reward'],
        'r_target': DEFAULT_CONFIG['r_target'],
        'num_drones': len(dictionary_context['drones']),
        'drone_positions': dictionary_context['drones'],
        'target_pos': dictionary_context['target'],
        'map_type': dictionary_context['map_type'],
        'peaks': dictionary_context['peaks'],
        'traces': dictionary_context['traces'],
        'obstacles': dictionary_context['obstacles']
    }
    
    # Precompute BFS distances
    obstacle_map = initialize_obstacle_map(params_dict)
    dist_BFS = precompute_BFS_distances(map_size, obstacle_map)
    
    params_dict['dist_BFS'] = dist_BFS      # Add the BFS distance lookup table to parameters
    
    # Call DARP partitioning if uniform map is selected, to divide the map into equal areas for each drone
    if params_dict['map_type'] == 1: 
        print("\n[DARP] Starting DARP algorithm...")
        assignment_matrix = darp_partitioning(params_dict)
        params_dict['darp_assignment'] = assignment_matrix
        print("[DARP] Division completed successfully.")

    # return the complete parameters dictionary
    return params_dict


def initialize_obstacle_map(params):
    map_size = params['map_size']
    obstacle_map = np.zeros((map_size, map_size), dtype=int)
    for obs_pos in params.get('obstacles', []):
        obstacle_map[obs_pos[0], obs_pos[1]] = 1
    return obstacle_map

def initialize_belief_map(params):
    map_size = params['map_size']
    map_type = params['map_type']
    peaks = params.get('peaks', [])
    belief_map = np.zeros((map_size, map_size))
    
    if map_type == 1:
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

    if 'obstacles' in params and len(params['obstacles']) > 0:
        obstacle_map = initialize_obstacle_map(params)
        belief_map = belief_map * (1 - obstacle_map)
    
    sum_prob = np.sum(belief_map)
    if sum_prob == 0:
        belief_map.fill(1.0 / (map_size * map_size))
    else:
        belief_map /= sum_prob
    return belief_map


# Function to perform DARP partitioning to divide areas in equal parts for each drone
def darp_partitioning(params, max_iter=80000, variate_weight=0.01, random_level=0.0001, limit_cells_diff=2, use_importance=True):
    
    map_size = params['map_size']
    num_drones = params['num_drones']
    drone_positions = params['drone_positions']
    obstacle_map = initialize_obstacle_map(params) 
    
    # Parameters and initial calculations
    tot_obstacles = np.sum(obstacle_map) 
    tot_cells = map_size * map_size 
    free_cells = tot_cells - num_drones - tot_obstacles

    term_thr = 1 if free_cells % num_drones != 0 else 0     # Starting threshold for termination condition, 0 if perfect division is possible, otherwise 1

    # Initialization matrices D based on BFS distances for each drone
    list_mat_D = np.zeros((num_drones, map_size, map_size))
    for r in range(num_drones):
        start_pos = drone_positions[r]
        for i in range(map_size):
            for j in range(map_size):
                if obstacle_map[i, j] == 1 or (i, j) in drone_positions:
                    list_mat_D[r, i, j] = float('inf') # Ignore obstacles and drone initial positions
                else:
                    dist = params['dist_BFS'].get((start_pos, (i, j)))
                    list_mat_D[r, i, j] = dist if dist is not None else float('inf')
                    
    # Management of previous infinite values for subsequent calculations (we replace inf values with a large finite values related to the maximum valid distance found)
    max_valid_dist = np.max(list_mat_D[list_mat_D != float('inf')]) 
    list_mat_D[list_mat_D == float('inf')] = max_valid_dist * 2 # Penalizza celle irraggiungibili
    
    # Calculate importance of cells based on distances (cells closer to a drone and farther from others are more important) 
    cells_importance = np.zeros((num_drones, map_size, map_size))
    max_importance = np.zeros(num_drones)
    min_importance = np.full(num_drones, float('inf'))
    
    # Calculation of importance values for each cell and drone, based on the formula: Importance = 1 / (Sum_Other_Distances) 
    for i in range(map_size):
        for j in range(map_size):
            tot_dist_sum = np.sum(list_mat_D[:, i, j]) 
            for r in range(num_drones):
                other_dist_sum = tot_dist_sum - list_mat_D[r, i, j]
                cells_importance[r, i, j] = 1.0 / other_dist_sum if other_dist_sum > 0 else 0.0
                
                if cells_importance[r, i, j] > max_importance[r]:
                    max_importance[r] = cells_importance[r, i, j]
                if cells_importance[r, i, j] < min_importance[r]:
                    min_importance[r] = cells_importance[r, i, j]

    # Initialize assignment matrix A, list of assigned cells for each drone, and connectivity status of regions for each drone 
    list_mat_D_copy = np.copy(list_mat_D)
    mat_A = np.zeros((map_size, map_size), dtype=int)
    list_uavs_cells = np.zeros(num_drones, dtype=int)
    list_connected_regions = np.zeros(num_drones, dtype=bool)
    
    success = False     # Termination flag
    
    # DARP logic algorithm 

    # While cycle to increment the threshold for termination condition (max tolerance=2)
    while term_thr <= limit_cells_diff and not success:
        
        down_thres = (tot_cells - term_thr * (num_drones - 1)) / (tot_cells * num_drones)
        upper_thres = (tot_cells + term_thr) / (tot_cells * num_drones)
        
        success = True
        iter_count = 0
        
        # Main loop of DARP algorithm
        while iter_count <= max_iter:
            
            # Assignment A matrix based on the minimum distance metric D and count of assigned cells for each drone 
            list_uavs_cells.fill(0)
            list_personal_assignment = np.zeros((num_drones, map_size, map_size), dtype=int)
            
            for i in range(map_size):
                for j in range(map_size):
                    if obstacle_map[i, j] == 0 and (i, j) not in drone_positions:
                        ind_min = np.argmin(list_mat_D_copy[:, i, j])
                        mat_A[i, j] = ind_min
                        list_personal_assignment[ind_min, i, j] = 1
                        list_uavs_cells[ind_min] += 1
                    else:
                        mat_A[i, j] = -1 # Obstacles/drone positions

            # Assicura le posizioni iniziali
            for r, pos in enumerate(drone_positions):
                list_personal_assignment[r, pos[0], pos[1]] = 1
                
            list_mat_C = []
            plainErrors = np.zeros(num_drones)      # Percentage of cells assigned to each drone (for fairness evaluation)
            divFairError = np.zeros(num_drones)     # Deviation from the ideal division (for fairness evaluation)
            
            # Check connectivity of assigned regions for each drone
            for r in range(num_drones):
                normalized_mat_C = np.ones((map_size, map_size))
                list_connected_regions[r] = True
                
                # Adjacent cells condition (up, down, left, right)
                adj_cells = np.array([[0,1,0],
                                      [1,1,1],
                                      [0,1,0]]) 
                labeled_array, num_islands = label(list_personal_assignment[r], structure=adj_cells)
                
                if num_islands > 1:
                    list_connected_regions[r] = False
                    
                    # Find initial position label for the drone's region
                    start_label = labeled_array[drone_positions[r][0], drone_positions[r][1]]
                    
                    # Region Ri: connected cells of the drone 
                    Ri_reg = (labeled_array == start_label).astype(int)
                    # Region Qi: disconnected cells of the drone (islands)
                    Qi_reg = ((list_personal_assignment[r] == 1) & (labeled_array != start_label)).astype(int)
                    
                    # Formula of connectivity matrix: C_i|x,y= min(||[x,y]−r||) − min(||[x,y]−q||)
                    dist_to_uav = distance_transform_edt(1 - Ri_reg)        # min(||[x,y]−r||)
                    dist_to_island = distance_transform_edt(1 - Qi_reg)     # min(||[x,y]−q||)
                    
                    mat_C = dist_to_uav - dist_to_island
                    max_v, min_v = np.max(mat_C), np.min(mat_C)
                    if max_v > min_v:
                        normalized_mat_C = (mat_C - min_v) * ((2 * variate_weight) / (max_v - min_v)) + (1 - variate_weight)
                
                list_mat_C.append(normalized_mat_C)
                
                # Division error evaluation for fairness (same number of cells assigned to each drone)
                plainErrors[r] = list_uavs_cells[r] / free_cells
                if plainErrors[r] < down_thres:
                    divFairError[r] = down_thres - plainErrors[r]
                elif plainErrors[r] > upper_thres:
                    divFairError[r] = upper_thres - plainErrors[r]
            
            # Termination condition 
            max_cells_ass = np.max(list_uavs_cells)
            min_cells_ass = np.min(list_uavs_cells)
            if (max_cells_ass - min_cells_ass) <= term_thr and np.all(list_connected_regions):
                break
                 
            total_neg_perc = np.sum(np.abs(divFairError[divFairError < 0])) # Total percentage of negative fairness errors
            total_neg_plain_errors = np.sum(plainErrors[divFairError < 0])  # Total sum of plain errors for negative fairness errors
            
            # Normalized formula for fairness correction coefficient m (basic formula: mi =mi +c(ki −f))
            for r in range(num_drones):
                coeff_m = 1.0
                if total_neg_plain_errors != 0.0:
                    if divFairError[r] < 0.0:
                        coeff_m = 1.0 + (plainErrors[r] / total_neg_plain_errors) * (total_neg_perc / 2.0)
                    else:
                        coeff_m = 1.0 - (plainErrors[r] / total_neg_plain_errors) * (total_neg_perc / 2.0)
                
                # Union of coeff_m and connectivity information to find final correction
                criterionMatrix = np.copy(cells_importance[r])
                if use_importance:
                    diff_imp = max_importance[r] - min_importance[r]
                    if divFairError[r] < 0:
                        criterionMatrix = (cells_importance[r] - min_importance[r]) * ((coeff_m - 1) / diff_imp) + 1
                    else:
                        criterionMatrix = (cells_importance[r] - min_importance[r]) * ((1 - coeff_m) / diff_imp) + coeff_m
                else:
                    criterionMatrix.fill(coeff_m)
                    
                # Random variation to manage the local minima and pairs of cells with the same distance values
                RM = 2.0 * random_level * np.random.rand(map_size, map_size) + 1.0 - random_level
                
                # Final update formula : Ei =Ci ⦿ (miEi)
                list_mat_D_copy[r] = list_mat_D_copy[r] * criterionMatrix * RM * list_mat_C[r]

            iter_count += 1
            
        if iter_count >= max_iter:
            max_iter //= 2
            success = False
            term_thr += 1

    return mat_A

# BFS algorithm to precompute distances between cells, considering obstacles (used in rollout)
def precompute_BFS_distances(map_size, obstacle_map):

    dist_BFS = {}
    
    # Movement directions (excluding 'Stay' because we search for the shortest path)
    directions = [delta for action, delta in MOVES_DELTA.items() if action != 'Stay']
    
    # Iterate over each cell as starting point
    for start_r in range(map_size):
        for start_c in range(map_size):
            start_pos = (start_r, start_c)
            
            # Skip cells that are obstacles
            if obstacle_map[start_r, start_c] == 1:
                continue
            
            # Positions to explore are placed in the queue, along with current distance from start_pos
            queue = deque([(start_pos, 0)])
            visited = {start_pos}
            
            while queue:
                current_pos, dist = queue.popleft()     # Remove and return the first position in the queue
                
                # Save the distance in the lookup table
                dist_BFS[(start_pos, current_pos)] = dist
                
                # Explore neighbors
                for dr, dc in directions:
                    next_r = current_pos[0] + dr
                    next_c = current_pos[1] + dc
                    next_pos = (next_r, next_c)
                    
                    # Check map limits
                    if not (0 <= next_r < map_size and 0 <= next_c < map_size):
                        continue
                    
                    # Skip obstacles
                    if obstacle_map[next_r, next_c] == 1:
                        continue
                    
                    # Skip if already visited
                    if next_pos in visited:
                        continue
                    
                    visited.add(next_pos)               # Cell marked as visited
                    queue.append((next_pos, dist + 1))  # Cell added to those to explore
    
    return dist_BFS


# =============================================================================
# 2. CENTRALIZED POMCP LOGIC 
# =============================================================================

class POMCPNode:
    def __init__(self, belief_map, parent=None):
        self.belief_map = belief_map
        self.parent = parent
        self.total_node_visits = 0
        self.children = {}
        self.q_value_actions = {}
        self.action_counts = {}

    def is_leaf(self):
        return self.total_node_visits == 0

class POMCPSolver:
    def __init__(self, params, obstacle_map, dist_BFS):
        self.params = params
        self.map_size = params['map_size']
        self.obstacle_map = obstacle_map
        self.dist_BFS = dist_BFS
        self.num_drones = params['num_drones']
        
        self.max_time = params['max_time']
        self.depth_limit = params['depth_limit']
        self.gamma = params['discount_factor']
        self.c = params['exploration_const']
        self.sensor_alpha = params['alpha_sensor']
        self.sensor_beta = params['beta_sensor']
        self.reward_alpha = params['reward_alpha']

    def search(self, current_belief_map, drone_positions, current_visited_cells):
        self.total_nodes_created = 1
        self.max_depth_reached = 0
        root = POMCPNode(belief_map=current_belief_map, parent=None)
        self.root = root 
        start_time = time.time()
        
        while True:
            if (time.time() - start_time) > self.max_time:
                break
            
            sampled_target_pos = self._sample_target_from_belief(root.belief_map)
            # Stato centrale: Target e tupla delle posizioni di tutti i droni
            state = (sampled_target_pos, tuple(drone_positions))
            self.simulate(state, root, 0, visited_cells=None, current_visited_cells=current_visited_cells)
        
        best_action = self._select_best_action(root)
        return best_action

    def simulate(self, state, node, depth, visited_cells=None, current_visited_cells=None):
        if visited_cells is None:
            visited_cells = set()
        else:
            visited_cells = visited_cells.copy()
            
        current_visited_cells = current_visited_cells.copy()

        if depth > self.max_depth_reached:
            self.max_depth_reached = depth

        if depth >= self.depth_limit:
            return 0.0

        if node.is_leaf():
            self.expand(node, state)
            if not node.action_counts: # Stallo totale
                return -100.0
            rollout_value = self.rollout(state)
            node.total_node_visits += 1
            return rollout_value

        action = self._ucb_search(node)
        next_state, observation, reward, terminal = self.generative_model_G(state, action, node.belief_map, visited_cells, current_visited_cells)

        if (action, observation) in node.children:
            child_node = node.children[(action, observation)]
        else:
            _, next_drones_pos = next_state
            # Aggiornamento bayesiano cumulativo per l'osservazione congiunta
            new_belief = node.belief_map.copy()
            for i, obs in enumerate(observation):
                new_belief = get_updated_belief_map_with_sensors(new_belief, next_drones_pos[i], obs, self.sensor_alpha, self.sensor_beta, self.obstacle_map)
            
            child_node = POMCPNode(belief_map=new_belief, parent=node)
            node.children[(action, observation)] = child_node
            self.total_nodes_created += 1

        if terminal:
            future_reward = 0.0
        else:
            future_reward = self.simulate(next_state, child_node, depth + 1, visited_cells, current_visited_cells)

        q_value = reward + self.gamma * future_reward
        node.total_node_visits += 1
        node.action_counts[action] += 1
        old_q = node.q_value_actions[action]        
        node.q_value_actions[action] = old_q + (q_value - old_q) / node.action_counts[action]

        return q_value

    def expand(self, node, state):
        _, drones_pos = state
        actions = list(MOVES_DELTA.keys())
        
        # Genera azioni congiunte
        for joint_action in itertools.product(actions, repeat=self.num_drones):
            next_positions = []
            valid = True
            
            for i, act in enumerate(joint_action):
                delta = MOVES_DELTA[act]
                curr_pos = drones_pos[i]
                nxt = (curr_pos[0] + delta[0], curr_pos[1] + delta[1])
                
                if not (0 <= nxt[0] < self.map_size and 0 <= nxt[1] < self.map_size):
                    valid = False; break
                if self.obstacle_map[nxt[0], nxt[1]] == 1:
                    valid = False; break
                next_positions.append(nxt)
            
            if not valid: continue
            
            # Controllo collisioni (2 droni nella stessa cella)
            if len(set(next_positions)) != self.num_drones:
                continue
                
            # Controllo swap
            swap = False
            for i in range(self.num_drones):
                for j in range(i+1, self.num_drones):
                    if next_positions[i] == drones_pos[j] and next_positions[j] == drones_pos[i]:
                        swap = True
            if swap: continue

            if joint_action not in node.action_counts:
                node.action_counts[joint_action] = 0
                node.q_value_actions[joint_action] = 0.0

    def rollout(self, state):
        target_pos, drones_pos = state
        
        # 1. Calcoliamo tutte le distanze reali tramite BFS
        distances = []
        for pos in drones_pos:
            dist = self.dist_BFS.get((pos, target_pos), float('inf'))
            distances.append(dist)
            
        score = 0.0

        for dist in distances:
            if dist == float('inf'):
                break
            
            score += (self.gamma ** dist)
            
        return score

    def generative_model_G(self, state, joint_action, belief_map, visited_cells, current_visited_cells):
        target_pos, drones_pos = state
        next_drones_pos = []
        joint_obs = []
        terminal = False
        r_target_reward = 0.0
        r_token = 0.0
        explorative_bonus = 0.0

        for i, act in enumerate(joint_action):
            delta = MOVES_DELTA[act]
            nxt = (drones_pos[i][0] + delta[0], drones_pos[i][1] + delta[1])
            next_drones_pos.append(nxt)

            if nxt == target_pos:
                obs = 0 if np.random.rand() < self.sensor_beta else 1
            else:
                obs = 1 if np.random.rand() < self.sensor_alpha else 0
            joint_obs.append(obs)

            if nxt == target_pos and obs == 1:
                r_target_reward = self.params['r_target']
                terminal = True
            
            if nxt not in visited_cells:
                r_token += belief_map[nxt]
                visited_cells.add(nxt)
            
            if nxt not in current_visited_cells:
                explorative_bonus += self.params['explorative_reward']
                current_visited_cells.add(nxt)

        next_state = (target_pos, tuple(next_drones_pos))
        total_reward = r_target_reward + (self.reward_alpha * r_token) + explorative_bonus
        return next_state, tuple(joint_obs), total_reward, terminal

    def _sample_target_from_belief(self, belief_map):
        flat_probs = belief_map.flatten()
        total = np.sum(flat_probs)
        if abs(total - 1.0) > 1e-6:
            flat_probs = flat_probs / total if total > 1e-9 else np.ones_like(flat_probs) / flat_probs.size
        sampled_idx = np.random.choice(np.arange(belief_map.size), p=flat_probs)
        return np.unravel_index(sampled_idx, belief_map.shape)

    def _ucb_search(self, node):
        best_val = -float('inf')
        ucb_best_action = None 
        log_total_visits = math.log(node.total_node_visits) if node.total_node_visits > 0 else 0 
        infinite_actions = []

        for action, n_ba in node.action_counts.items():
            if n_ba == 0:
                infinite_actions.append(action)
            else:
                q_ba = node.q_value_actions[action]
                uct_val = q_ba + self.c * math.sqrt(log_total_visits / n_ba)
                if uct_val > best_val:
                    best_val = uct_val
                    ucb_best_action = action

        if infinite_actions:
            return random.choice(infinite_actions)
        return ucb_best_action

    def _select_best_action(self, node):
        if not node.q_value_actions:
            return tuple(['Stay'] * self.num_drones)
        return max(node.q_value_actions.items(), key=lambda x: x[1])[0]


# =============================================================================
# 3. CENTRALIZED BELIEF FUSION & AGENTS
# =============================================================================

def apply_trace_distribution(belief_map, trace_obs, params, obstacle_map):
    trace_type = trace_obs['type']
    trace_pos = trace_obs['pos']
    trace_params = trace_obs['trace_params']
    map_size = params['map_size']
    
    r, c = np.indices((map_size, map_size))
    delta_r = r - trace_pos[0]
    delta_c = c - trace_pos[1]

    if trace_type == 'von_mises':
        angles = np.arctan2(delta_r, delta_c)
        diff = angles - trace_params['mu']
        trace_distribution = np.exp(trace_params['kappa'] * np.cos(diff)) / (2 * np.pi * np.i0(trace_params['kappa']))
        trace_distribution[trace_pos[0], trace_pos[1]] = 1.0
    elif trace_type == 'ring':
        dist_matrix = np.sqrt(delta_r**2 + delta_c**2)
        trace_distribution = np.exp(-((dist_matrix - trace_params['radius'])**2) / (2 * trace_params['variance']))
    elif trace_type == 'gaussian':
        x, y = np.mgrid[0:map_size, 0:map_size]
        cov_matrix = [[trace_params['sigma_x']**2, 0], [0, trace_params['sigma_y']**2]]
        rv = multivariate_normal(trace_pos, cov_matrix)
        trace_distribution = rv.pdf(np.dstack((x, y)))

    fused_belief = belief_map * trace_distribution
    fused_belief = fused_belief * (1 - obstacle_map)
    total_prob = np.sum(fused_belief)
    
    if total_prob > 1e-9:
        fused_belief /= total_prob
    else:
        fused_belief = belief_map.copy()
    return fused_belief

def get_updated_belief_map_with_sensors(current_belief, pos, obs, alpha, beta, obstacle_map):
    Psi = 1.0 - beta if obs == 1 else beta
    Phi = alpha if obs == 1 else 1.0 - alpha
    Omega = Psi - Phi
    p_st = current_belief[pos]
    Z = Phi + (Omega * p_st)

    if Z < 1e-9:
        return current_belief 

    new_belief = (current_belief.copy() * Phi) / Z 
    new_belief[pos] = (Psi * p_st) / Z
    new_belief = new_belief * (1 - obstacle_map)
    total = np.sum(new_belief)
    
    if total > 1e-9:
        new_belief /= total
    return new_belief


class DroneAgent:
    def __init__(self, drone_id, start_pos, params):
        self.id = drone_id
        self.pos = start_pos
        self.params = params
        self.observation = None
        self.positive_obs_count = 0
        self.tsp_plan = deque()

    def execute_move(self, action):
        d = MOVES_DELTA.get(action, (0, 0))
        self.pos = (self.pos[0] + d[0], self.pos[1] + d[1])

    def get_real_observation(self, target_pos, traces, discovered_traces):
        trace_found = None
        for trace in traces:
            if trace['pos'] == self.pos and self.pos not in discovered_traces:
                trace_found = trace
                break
        
        if trace_found:
            if np.random.rand() < self.params['beta_sensor']:
                self.observation = 0
            else:
                self.observation = trace_found
        else:
            if self.pos == target_pos:
                self.observation = 0 if np.random.rand() < self.params['beta_sensor'] else 1
            else:
                self.observation = 1 if np.random.rand() < self.params['alpha_sensor'] else 0


# TSP solver for systematic exploration of assigned DARP areas.
class TSPSolver:
    
    def __init__(self, map_size, obstacle_map, drone_id, start_pos, darp_matrix):
        self.map_size = map_size
        self.obstacle_map = obstacle_map
        self.drone_id = drone_id
        self.start_pos = start_pos
        self.darp_matrix = darp_matrix
    
    # Generate a sequence of actions to visit all cells assigned to the drone by DARP
    def generate_full_plan(self):
        local_obstacle_map = np.copy(self.obstacle_map)     # Local map where non-DARP cells are treated as obstacles
        free_cells = []
        
        for r in range(self.map_size):
            for c in range(self.map_size):
                
                if self.darp_matrix[r, c] == self.drone_id - 1 and self.obstacle_map[r, c] == 0:
                    free_cells.append((r, c))
                else:
                    local_obstacle_map[r, c] = 1
                    
        # INUTILE??
        if self.start_pos not in free_cells:
            free_cells.insert(0, self.start_pos)
            local_obstacle_map[self.start_pos[0], self.start_pos[1]] = 0
            
        num_free_cells = len(free_cells)

        # INUTILE??
        if num_free_cells <= 1:
            return []
            
        bfs_distances = precompute_BFS_distances(self.map_size, local_obstacle_map)
                
        elk_matrix = [[0] * num_free_cells for _ in range(num_free_cells)]
        for i in range(num_free_cells):
            for j in range(num_free_cells):
                pos_i = free_cells[i]
                pos_j = free_cells[j]
                if i == j:
                    elk_matrix[i][j] = 0
                else:
                    elk_matrix[i][j] = int(bfs_distances.get((pos_i, pos_j), 999999))
        
        # Use elkai to find the TSP tour
        try:
            tour_indices = elkai.solve_int_matrix(elk_matrix)
        except Exception as e:
            print(f"  [D{self.drone_id}] Error in elkai solver: {e}")
            return []
            
        # Reorder the tour to start from the drone's starting position
        start_idx = free_cells.index(self.start_pos)
        if start_idx in tour_indices:
            idx_in_tour = tour_indices.index(start_idx)
            ordered_tour = tour_indices[idx_in_tour:] + tour_indices[:idx_in_tour]
        else:
            ordered_tour = tour_indices
            
        # Add the start position at the end of the tour to complete the cycle
        ordered_tour.append(ordered_tour[0])
            
        # Convert sequence of nodes (cells) to sequence of actions (N, S, E, W)
        actions = []
        current_pos = free_cells[ordered_tour[0]]
        
        for next_node_idx in ordered_tour[1:]:
            next_pos = free_cells[next_node_idx]
            
            # Construct path from current_pos to next_pos using steepest descent on BFS distances from next_pos
            while current_pos != next_pos:
                best_action = None
                best_next = None
                min_d = float('inf')
                
                for action, delta in MOVES_DELTA.items():
                    if action == 'Stay':
                        continue
                    nr, nc = current_pos[0] + delta[0], current_pos[1] + delta[1]
                    if 0 <= nr < self.map_size and 0 <= nc < self.map_size and local_obstacle_map[nr, nc] == 0:
                        d = bfs_distances.get((next_pos, (nr, nc)), float('inf'))
                        if d < min_d:
                            min_d = d
                            best_action = action
                            best_next = (nr, nc)
                            
                if best_action is None:
                    break  # Edge case: disconnected region, skip to next node
                    
                actions.append(best_action)
                current_pos = best_next
                
        return actions


# =============================================================================
# 4. GRAPHIC FUNCTIONS 
# =============================================================================

def init_graphics(params):
    pygame.init()
    map_size = params['map_size']
    display_info = pygame.display.Info()
    
    sidebar_w = 400
    cell_size = min((int(display_info.current_w * 0.90) - sidebar_w) // map_size, int(display_info.current_h * 0.90) // map_size)

    GRID_WIDTH = map_size * cell_size
    screen_w = min(GRID_WIDTH + sidebar_w, int(display_info.current_w * 0.90))
    screen_h = min(map_size * cell_size, int(display_info.current_h * 0.90))
    
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Multi-Drone POMCP Centralizzato")

    spacing = max(2, min(25, int((screen_h - 101) / (params['num_drones'] * 7))))
    font_sidebar_size = max(4, min(25, int(spacing * 0.8)))

    return {
        'screen': screen,
        'background_surface': pygame.Surface((GRID_WIDTH, screen_h)),
        'font_cell': pygame.font.SysFont(None, max(1, cell_size // 3)),
        'font_sidebar': pygame.font.SysFont(None, font_sidebar_size),
        'font_sidebar_fixed': pygame.font.SysFont(None, 16),
        'clock': pygame.time.Clock(),
        'GRID_WIDTH': GRID_WIDTH,
        'CELL_SIZE': cell_size,
        'SIDEBAR_WIDTH': sidebar_w,
        'params': params,
        'spacing': spacing
    }

def draw_static_background(graphics_ctx, global_belief_map, drones):
    surface = graphics_ctx['background_surface']
    cell_size = graphics_ctx['CELL_SIZE']
    font_cell = graphics_ctx['font_cell']
    params = graphics_ctx['params']
    map_size = params['map_size']
    
    max_prob = global_belief_map.max()
    obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else None
    
    darp_matrix = params.get('darp_assignment', None)
    drone_start_positions = params.get('drone_positions', [])
    drone_colors = DARP_AREA_COLORS

    surface.fill(COLORS['WHITE'])

    # Controlliamo se siamo in modalità POMCP (se il solver è attivo, togliamo i colori DARP)
    is_pomcp_mode = any(getattr(drone, 'tsp_plan', None) == [] for drone in drones) # Semplificazione

    for r in range(map_size):
        for c in range(map_size):
            x = c * cell_size
            y = r * cell_size
            prob = global_belief_map[r, c]

            if obstacle_map is not None and obstacle_map[r, c] == 1:
                color = COLORS['BLACK']
            else:
                if not is_pomcp_mode and darp_matrix is not None and (darp_matrix[r, c] != -1 or (r, c) in drone_start_positions):
                    if darp_matrix[r, c] != -1:
                        drone_idx = darp_matrix[r, c]
                    else:
                        drone_idx = drone_start_positions.index((r, c))
                        
                    base_color = drone_colors[drone_idx % len(drone_colors)]
                    
                    if max_prob > 1e-9:
                        intensity = ((prob / max_prob) ** 0.4)
                        color = (
                            int(255 - (255 - base_color[0]) * (0.2 + 0.6 * intensity)),
                            int(255 - (255 - base_color[1]) * (0.2 + 0.6 * intensity)),
                            int(255 - (255 - base_color[2]) * (0.2 + 0.6 * intensity))
                        )
                    else:
                        color = (
                            int(255 - (255 - base_color[0]) * 0.2),
                            int(255 - (255 - base_color[1]) * 0.2),
                            int(255 - (255 - base_color[2]) * 0.2)
                        )
                else:
                    if max_prob > 1e-9:
                        color_val = int(255 * ((prob / max_prob) ** 0.4))
                        color = (255 - color_val, 255 - color_val, 255)
                    else:
                        color = (255, 255, 255)

            pygame.draw.rect(surface, color, (x, y, cell_size, cell_size))
            pygame.draw.rect(surface, COLORS['BLACK'], (x, y, cell_size, cell_size), 1)

            if obstacle_map is None or obstacle_map[r, c] == 0:
                text = font_cell.render(f"{prob * 100:.3f}%", True, COLORS['BLACK'])
                text_rect = text.get_rect(centerx=x + cell_size // 2, bottom=y + cell_size - max(2, cell_size // 20))
                surface.blit(text, text_rect)

def draw_elements(graphics_ctx, global_belief_map, drones, target_pos, traces, stats):
    screen = graphics_ctx['screen']
    CELL_SIZE = graphics_ctx['CELL_SIZE']
    GRID_WIDTH = graphics_ctx['GRID_WIDTH']
    SIDEBAR_WIDTH = graphics_ctx['SIDEBAR_WIDTH']
    font_cell = graphics_ctx['font_cell']
    font_sidebar = graphics_ctx['font_sidebar']
    font_sidebar_fixed = graphics_ctx['font_sidebar_fixed']
    spacing = graphics_ctx['spacing']
    
    # Target
    tx, ty = target_pos
    target_rect = pygame.Rect(ty * CELL_SIZE, tx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.line(screen, COLORS['RED'], target_rect.topleft, target_rect.bottomright, 3)
    pygame.draw.line(screen, COLORS['RED'], target_rect.topright, target_rect.bottomleft, 3)

    # Traces
    for trace in traces:
        trace_r, trace_c = trace['pos']
        trace_square = pygame.Rect(trace_c * CELL_SIZE + CELL_SIZE // 4, trace_r * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(screen, COLORS['ORANGE'], trace_square, 0)
        center = (trace_c * CELL_SIZE + CELL_SIZE // 2, trace_r * CELL_SIZE + CELL_SIZE // 2)
        trace_label = font_cell.render("Tr", True, COLORS['WHITE'])
        screen.blit(trace_label, trace_label.get_rect(center=center))

    # Drones
    drone_colors = list(COLORS.values())[8:] # Colori distinti
    for drone in drones:
        dr, dc = drone.pos
        center = (dc * CELL_SIZE + CELL_SIZE // 2, dr * CELL_SIZE + CELL_SIZE // 2)
        color = drone_colors[(drone.id - 1) % len(drone_colors)]
        pygame.draw.circle(screen, color, center, CELL_SIZE // 3, 4)
        id_text = font_cell.render(str(drone.id), True, color)
        screen.blit(id_text, id_text.get_rect(center=center))

    # Sidebar
    sidebar_rect = pygame.Rect(GRID_WIDTH, 0, SIDEBAR_WIDTH, screen.get_height())
    pygame.draw.rect(screen, COLORS['GRAY'], sidebar_rect)

    y_offset = 10
    for drone in drones:
        color = drone_colors[(drone.id - 1) % len(drone_colors)]
        screen.blit(font_sidebar.render(f"=== Drone {drone.id} ===", True, color), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing
        screen.blit(font_sidebar.render(f"Obs: {stats['drones'][drone.id].get('obs', '-')}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing
        screen.blit(font_sidebar.render(f"Sims (Root): {stats['drones'][drone.id].get('visits', 0)}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing
        screen.blit(font_sidebar.render(f"Max Depth: {stats['drones'][drone.id].get('depth', 0)}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing
        screen.blit(font_sidebar.render(f"Action: {stats['drones'][drone.id].get('best', '-')}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing + 10

    # Elementi fissi in basso
    screen_height = screen.get_height()
    controls_y = screen_height - 26
    screen.blit(font_sidebar_fixed.render("SPAZIO: Avvia/Pausa  |  R: Riavvia  |  ESC: Esci", True, COLORS['BLACK']), (GRID_WIDTH + 10, controls_y))

    bar_y = controls_y - 33
    bar_width = SIDEBAR_WIDTH - 40
    max_prob = global_belief_map.max()
    pygame.draw.rect(screen, COLORS['WHITE'], (GRID_WIDTH + 20, bar_y, bar_width, 18))
    pygame.draw.rect(screen, COLORS['LIGHT_BLUE'], (GRID_WIDTH + 20, bar_y, bar_width * min(max_prob, 1.0), 18))
    
    pygame.draw.line(screen, COLORS['RED'], ((GRID_WIDTH + 20) + bar_width * 0.95, bar_y - 2), ((GRID_WIDTH + 20) + bar_width * 0.95, bar_y + 20), 2)
    
    max_prob_y = bar_y - 26
    screen.blit(font_sidebar_fixed.render(f"Max Prob: {max_prob * 100:.2f}%", True, COLORS['BLACK']), (GRID_WIDTH + 20, max_prob_y))
    text_thr = font_sidebar_fixed.render("Threshold: 95%", True, COLORS['RED'])
    screen.blit(text_thr, (GRID_WIDTH + SIDEBAR_WIDTH - 20 - text_thr.get_width(), max_prob_y))
    screen.blit(font_sidebar_fixed.render(f"Step: {stats['step']}", True, COLORS['BLACK']), (GRID_WIDTH + 20, max_prob_y - 16))

def draw_tsp_paths(graphics_ctx, drones):
    # La funzione draw_tsp_paths rimane invariata
    screen = graphics_ctx['screen']
    CELL_SIZE = graphics_ctx['CELL_SIZE']
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    def draw_boundary_arrow(surface, color, start_pos, end_pos, width=1, is_double=False):
        x1, y1 = start_pos; x2, y2 = end_pos
        pygame.draw.line(surface, color, (x1, y1), (x2, y2), width)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = max(5, CELL_SIZE * 0.20)
        arrow_rad = 0.5 
        
        def draw_head(cx, cy, ang):
            p1 = (cx - arrow_len * math.cos(ang - arrow_rad), cy - arrow_len * math.sin(ang - arrow_rad))
            p2 = (cx - arrow_len * math.cos(ang + arrow_rad), cy - arrow_len * math.sin(ang + arrow_rad))
            pygame.draw.line(surface, color, (cx, cy), p1, width)
            pygame.draw.line(surface, color, (cx, cy), p2, width)
            
        draw_head(mx, my, angle)
        if is_double: draw_head(mx, my, angle + math.pi)

    from collections import Counter
    has_paths = False
    
    for drone in drones:
        if hasattr(drone, 'tsp_plan') and drone.tsp_plan:
            current_r, current_c = drone.pos
            path_cells = [(current_r, current_c)]
            
            for action in drone.tsp_plan:
                dr, dc = MOVES_DELTA.get(action, (0, 0))
                if action == 'Stay': continue
                current_r += dr; current_c += dc
                path_cells.append((current_r, current_c))
                
            if len(path_cells) < 2: continue
            has_paths = True
            
            cell_visits = Counter(path_cells)
            for i in range(len(path_cells) - 1):
                r1, c1 = path_cells[i]
                r2, c2 = path_cells[i+1]
                
                x1 = c1 * CELL_SIZE + CELL_SIZE // 2
                y1 = r1 * CELL_SIZE + CELL_SIZE // 2
                x2 = c2 * CELL_SIZE + CELL_SIZE // 2
                y2 = r2 * CELL_SIZE + CELL_SIZE // 2
                
                is_overlap = cell_visits[(r1, c1)] > 1 or cell_visits[(r2, c2)] > 1
                color = (255, 0, 0, 110) if is_overlap else (0, 0, 0, 110)
                draw_boundary_arrow(overlay, color, (x1, y1), (x2, y2), width=1, is_double=is_overlap)
                
    if has_paths: screen.blit(overlay, (0, 0))

def render_frame(graphics_ctx, global_belief_map, drones, target_pos, traces, ui_stats):
    draw_static_background(graphics_ctx, global_belief_map, drones)
    graphics_ctx['screen'].fill(COLORS['WHITE'])
    graphics_ctx['screen'].blit(graphics_ctx['background_surface'], (0, 0))
    
    draw_elements(graphics_ctx, global_belief_map, drones, target_pos, traces, ui_stats)
    draw_tsp_paths(graphics_ctx, drones)
    
    pygame.display.flip()


# =============================================================================
# 5. MAIN LOOP
# =============================================================================

def run_simulation(params):
    graphics_ctx = init_graphics(params)
    
    num_drones = params['num_drones']
    target_pos = params['target_pos']
    traces = params['traces']
    obstacle_map = initialize_obstacle_map(params)
    dist_BFS = params.get('dist_BFS', precompute_BFS_distances(params['map_size'], obstacle_map))
    
    global_belief_map = initialize_belief_map(params)
    global_discovered_traces = set()
    global_explored_cells = set()
    
    # Determina la modalità iniziale
    global_mode = 'TSP' if params['map_type'] == 1 else 'POMCP'
    
    drones = [DroneAgent(i+1, params['drone_positions'][i], params) for i in range(num_drones)]
    centralized_solver = POMCPSolver(params, obstacle_map, dist_BFS)

    # Inizializza i piani TSP se in modalità Uniforme
    if global_mode == 'TSP':
        for drone in drones:
            tsp = TSPSolver(params['map_size'], obstacle_map, drone.id, drone.pos, params.get('darp_assignment'))
            drone.tsp_plan.extend(tsp.generate_full_plan())

    running = True
    auto_mode = False
    step_counter = 0
    move_interval_sec = 1.0 # Velocità di esecuzione dei turni in Auto Mode
    last_step_time = 0.0
    
    while running:
        
        # ==========================================
        # ESECUZIONE STEP (LOGICA)
        # ==========================================
        if auto_mode and (time.monotonic() - last_step_time) >= move_interval_sec:
            step_counter += 1
            print(f"\n--- STEP {step_counter} | MODALITA': {global_mode} ---")
            
            # 1. Switch da TSP a POMCP se tutti i droni hanno finito il loro giro DARP
            if global_mode == 'TSP' and all(len(d.tsp_plan) == 0 for d in drones):
                global_mode = 'POMCP'
                print("Tutti i piani TSP completati. Ritorno alla ricerca centralizzata POMCP.")
            
            joint_action = []
            
            # 2. Pianificazione Azioni
            if global_mode == 'TSP':
                for drone in drones:
                    # Se ha visto il target o una traccia precedentemente, aspetta un turno
                    if drone.observation == 1:
                        drone.positive_obs_count += 1
                        drone.tsp_plan.appendleft('Stay')
                    elif drone.observation == 0 and drone.positive_obs_count > 0:
                        drone.positive_obs_count -= 1
                        if drone.positive_obs_count > 0:
                            drone.tsp_plan.appendleft('Stay')
                    
                    if drone.tsp_plan:
                        joint_action.append(drone.tsp_plan.popleft())
                    else:
                        joint_action.append('Stay')
            
            elif global_mode == 'POMCP':
                current_positions = [d.pos for d in drones]
                # Il solver centralizzato esplora l'albero e trova la joint action (tuple) migliore
                joint_action = centralized_solver.search(global_belief_map.copy(), current_positions, global_explored_cells)

            # 3. Esecuzione Fisica e Percezione Sensori
            for i, drone in enumerate(drones):
                drone.execute_move(joint_action[i])
                drone.get_real_observation(target_pos, traces, global_discovered_traces)
                global_explored_cells.add(drone.pos)

            # 4. Aggiornamento Centralizzato della Mappa
            for drone in drones:
                obs = drone.observation
                
                # Se l'osservazione è una traccia
                if isinstance(obs, dict) and 'type' in obs:
                    if obs['pos'] not in global_discovered_traces:
                        global_belief_map = apply_trace_distribution(global_belief_map, obs, params, obstacle_map)
                        global_discovered_traces.add(obs['pos'])
                        
                        # Forza tutti a passare in POMCP se una traccia viene scoperta durante il TSP
                        if global_mode == 'TSP':
                            global_mode = 'POMCP'
                            for d in drones: 
                                d.tsp_plan.clear()
                            print(f"[!] Traccia rilevata da Drone {drone.id}! Switch immediato a modalità POMCP.")
                
                # Se l'osservazione è un rilevamento del target (0 o 1)
                elif isinstance(obs, int):
                    global_belief_map = get_updated_belief_map_with_sensors(
                        global_belief_map, drone.pos, obs, params['alpha_sensor'], params['beta_sensor'], obstacle_map
                    )

            # 5. Condizione di Vittoria
            if global_belief_map.max() >= 0.95:
                print(f"\n🏆 TARGET TROVATO CON SUCCESSO! (Probabilità: {global_belief_map.max()*100:.1f}%)")
                auto_mode = False

            last_step_time = time.monotonic()

        # ==========================================
        # RENDERING E GESTIONE EVENTI
        # ==========================================
        
        # 1. Costruiamo le statistiche per la Sidebar
        ui_stats = {
            'step': step_counter,
            'drones': {}
        }
        
        for i, drone in enumerate(drones):
            # Identifichiamo l'azione corrente in base alla modalità
            current_action = '-'
            if global_mode == 'POMCP' and 'joint_action' in locals() and joint_action:
                current_action = joint_action[i]
            elif global_mode == 'TSP' and drone.tsp_plan:
                current_action = drone.tsp_plan[0] if len(drone.tsp_plan) > 0 else 'Stay'

            # Osservazione formattata (se è traccia mettiamo 'Tr', se è num mettiamo num)
            obs_str = '-'
            if drone.observation is not None:
                obs_str = 'Tr' if isinstance(drone.observation, dict) else str(drone.observation)

            ui_stats['drones'][drone.id] = {
                'obs': obs_str,
                'depth': getattr(centralized_solver, 'max_depth_reached', 0) if global_mode == 'POMCP' else 0,
                'visits': getattr(centralized_solver.root, 'total_node_visits', 0) if global_mode == 'POMCP' and hasattr(centralized_solver, 'root') else 0,
                'nodes': getattr(centralized_solver, 'total_nodes_created', 0) if global_mode == 'POMCP' else 0,
                'best': current_action
            }

        # 2. Rendering Frame aggiornato
        render_frame(graphics_ctx, global_belief_map, drones, target_pos, traces, ui_stats)
        graphics_ctx['clock'].tick(30)

        # 3. Input Tastiera Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: 
                    return "quit"
                if event.key == pygame.K_r: 
                    return "restart"
                if event.key == pygame.K_SPACE: 
                    auto_mode = not auto_mode
                    if auto_mode:
                        # Rimuoviamo il delay del timer per far scattare subito la prima mossa
                        last_step_time = time.monotonic() - move_interval_sec
                        print(f"\n▶ Modalità AUTO Avviata ({global_mode})")
                    else:
                        print("\n⏸ Modalità AUTO in Pausa")

# =============================================================================
# 5. MAIN ENTRY POINT
# =============================================================================

def main():
    while True:
        # 1. Configurazione iniziale tramite interfaccia grafica
        params = get_user_parameters()
        
        # 2. Avvio della simulazione vera e propria
        result = run_simulation(params)
        
        # 3. Gestione dell'esito (Riavvio o Uscita)
        if result == "quit":
            print("Simulazione terminata dall'utente.")
            pygame.quit()
            break  # Esce dal ciclo while e termina lo script
            
        elif result == "restart":
            print("Riavvio richiesto...")
            pygame.quit()
            # Il ciclo while riparte da capo, richiamando get_user_parameters()
            continue

if __name__ == "__main__":
    main()
