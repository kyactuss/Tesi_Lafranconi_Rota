import elkai
from collections import deque
import math
from scipy.ndimage import label, distance_transform_edt
import random
import time
import numpy as np
from scipy.stats import multivariate_normal
import pygame
import multiprocessing
from collections import deque

# =============================================================================
# 1.PARAMETERS CONFIGURATION 
# =============================================================================
DEFAULT_CONFIG = {
    'map_size': 20,
    'alpha_sensor': 0.03,
    'beta_sensor': 0.03,
    'max_time': 2,
    'depth_limit': 50,
    'discount_factor': 0.98,
    'exploration_const': math.sqrt(2),
    'reward_alpha': 3,
    'explorative_reward': 0.005,
    'r_target': 1,
}

# Movements related to actions, used for position updates
MOVES_DELTA = {
    'N': (-1, 0),
    'S': (1, 0),
    'W': (0, -1),
    'E': (0, 1),
    'Stay': (0, 0)
}

# Global color palette for all pygame graphics
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GRAY': (200, 200, 200),
    'LIGHT_GRAY': (240, 240, 240),
    'BLUE': (50, 100, 255),
    'DARK_BLUE': (0, 0, 200),
    'LIGHT_BLUE': (0, 0, 255),
    'RED': (255, 50, 50),
    'DARK_RED': (255, 0, 0),
    'LIGHT_RED': (200, 0, 0),
    'GREEN': (0, 200, 0),
    'DARK_GREEN': (0, 180, 0),
    'PURPLE': (200, 50, 255),
    'DARK_PURPLE': (100, 0, 100),
    'ORANGE': (255, 140, 0),
    'PINK': (255, 192, 203),
    'BROWN': (165, 42, 42),
    'CYAN': (0, 255, 255),
    'YELLOW': (255, 255, 0),
    'DARK_GRAY': (128, 128, 128)
}

# Fixed palette for DARP area rendering
DARP_AREA_COLORS = [
    (144, 238, 144),
    (255, 160, 160),
    (216, 191, 216),
    (173, 216, 230),
    (255, 255, 204),
    (210, 180, 140),
    (211, 211, 211),
    (255, 218, 185),
    (176, 224, 230),
    (221, 160, 221),
    (189, 252, 201),
    (255, 228, 196),
    (240, 230, 140),
    (224, 255, 255)
]

# Function to collect user input through graphical interface
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
                    exit()
                
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
                        exit()
        
        return selected
    
    #dictionary to store all selected parameters and positions during the configuration phases, to be passed as context to the selection function and for final return
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
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    dictionary_context['map_type'] = 1
                    waiting = False
                elif event.key == pygame.K_2:
                    dictionary_context['map_type'] = 2
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
    
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
    
    #Call DARP partitioning if uniform map is selected, to divide the map into equal areas for each drone
    if params_dict['map_type'] == 1: 
        print("\n[DARP] Starting DARP algorithm...")
        assignment_matrix = darp_partitioning(params_dict)
        params_dict['darp_assignment'] = assignment_matrix
        print("[DARP] Division completed successfully.")

    # return the complete parameters dictionary
    return params_dict


# Function to generate boolean obstacle map (1 = obstacle, 0 = free) 
def initialize_obstacle_map(params):
    
    map_size = params['map_size']
    obstacle_map = np.zeros((map_size, map_size), dtype=int)
    
    for obs_pos in params['obstacles']:
        r, c = obs_pos
        obstacle_map[r, c] = 1
    
    return obstacle_map


# function to initialize the belief map
def initialize_belief_map(params):

    map_size = params['map_size']
    map_type = params['map_type']
    peaks = params['peaks']
    
    # Initialize empty map
    belief_map = np.zeros((map_size, map_size))
    
    # Case 1: Uniform
    if map_type == 1:
        belief_map.fill(1.0) # Fill everything with 1 (we'll normalize later)
        
    # Case 2: Gaussians
    else:
        # Create grid and coordinates for pdf distribution
        x, y = np.mgrid[0:map_size, 0:map_size]
        coord = np.dstack((x, y))
        
        for peak in peaks:
            mean = peak['mean']     # (row, column)
            sigmas = peak['cov']    # [sx, sy]
            
            # Covariance matrix (diagonal for simplicity)
            cov_matrix = [[sigmas[0]**2, 0], [0, sigmas[1]**2]]
            
            # Create a continuous normal distribution with mean and cov_matrix
            rv = multivariate_normal(mean, cov_matrix)
                                    
            # Discretize the distribution on the grid and add to belief_map
            belief_map += rv.pdf(coord)

    # Apply obstacles before normalization
    if 'obstacles' in params and len(params['obstacles']) > 0:
        obstacle_map = initialize_obstacle_map(params)
        belief_map = belief_map * (1 - obstacle_map) #if obstacle_map[r,c] == 1 → belief_map[r,c] = 0, else unchanged
    
    # Map normalization
    sum_prob = np.sum(belief_map)
    
    if sum_prob == 0:
        # Safety fallback for huge variance values or too many obstacles
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
# 2. DEC-POMCP LOGIC 
# =============================================================================

class POMCPNode:
   
   # Constructor of the POMCP tree node
    def __init__(self, belief_map, parent=None):

        self.belief_map = belief_map    # Belief map associated with the node
        self.parent = parent            # Reference to the parent node (None for the root)
        self.total_node_visits = 0      # N(b): number of total visits to the node
        self.children = {}              # children key: (action, observation) -> reference to the child node
        self.q_value_actions = {}       # q_value_actions key: action -> value: Q(b, a) (Average value)
        self.action_counts = {}         # action_counts key: action -> value: N(b, a)

    # Check if the node is a leaf
    def is_leaf(self):
        return self.total_node_visits == 0

class POMCPSolver:
    def __init__(self, max_time, depth_limit, discount_factor, exploration_const, sensor_alpha, sensor_beta, reward_alpha, explorative_reward, map_size, obstacle_map, drone_id, dist_BFS, r_target, explored_cells):
    
        # Save configuration parameters as class attributes
        self.max_time = max_time
        self.depth_limit = depth_limit
        self.gamma = discount_factor
        self.c = exploration_const
        self.sensor_alpha = sensor_alpha
        self.sensor_beta = sensor_beta
        self.reward_alpha = reward_alpha
        self.explorative_reward = explorative_reward
        self.map_size = map_size
        self.obstacle_map = obstacle_map
        self.drone_id = drone_id
        self.r_target = r_target
        self.explored_cells = set(explored_cells)
        
        # Use precomputed BFS distances (optimization)
        self.dist_BFS = dist_BFS

    # POMCP search function: tree construction
    def search(self, current_belief_map, drone_position, partner_positions=None, partner_plans=None):

        self.drone_position = drone_position
        self.partner_positions = partner_positions
        self.partner_plans = partner_plans

        # Attributes for tree monitoring and statistics
        self.total_nodes_created = 1    # Count already the first node (the root) 
        self.max_depth_reached = 0

        # Creation of root node with current belief map
        root = POMCPNode(belief_map=current_belief_map, parent=None)
        self.root = root 
        
        start_time = time.time()    # Start timer for POMCP time limit
        

        # --- MAIN POMCP LOOP ---
        while True:
            if (time.time() - start_time) > self.max_time:
                break
            
            # Initial state sampling: extract target position at each iteration (state: drone pos and target pos)
            sampled_target_pos = self._sample_target_from_belief(root.belief_map)
            state = (sampled_target_pos, self.drone_position)
            
            # Start recursive simulation 
            self.simulate(state, root, 0, visited_cells=None, current_visited_cells=self.explored_cells)
        
        # Selection of best action
        best_action = self._select_best_action(root)
        
        # Extraction of future plans for all actions from root
        future_plans = self._extract_future_plans(root)
        
        # Print Q-values for all actions from root
        print(f"\n[Drone {self.drone_id}] Q-values from root:")
        for action, q_value in sorted(root.q_value_actions.items(), key=lambda x: x[1], reverse=True):
            visits = root.action_counts.get(action, 0)
            print(f"  Action '{action}': Q-value = {q_value:.4f}, Visits = {visits}")
        
        return best_action, future_plans

    # Recursive simulation function (executes expansion, rollout and backpropagation)
    def simulate(self, state, node, depth, visited_cells=None, current_visited_cells=None): 
        
        # Initialize the set at root for visited_cells
        if visited_cells is None:
            visited_cells = set()
        else:
            visited_cells = visited_cells.copy() # If it already exists, make a copy

        current_visited_cells = current_visited_cells.copy()

        # Update maximum depth reached for monitoring and statistics
        if depth > self.max_depth_reached:
            self.max_depth_reached = depth

        # Check if tree depth limit is reached
        if depth >= self.depth_limit:
            return 0.0

        # Check if node is leaf
        if node.is_leaf():

            # Node expansion
            is_root = (node.parent is None) # At root level, exclude moves to current positions of other drones
            self.expand(node, state, self.partner_positions if is_root else None)
            
            # Rollout
            rollout_value = self.rollout(state)

            node.total_node_visits += 1  # Count the visit to the just expanded leaf node
            
            return rollout_value

        # Action selection via UCT
        action = self._ucb_search(node)

        # Generative Model (G): simulation of state transition, observation and reward
        next_state, observation, reward, terminal = self.generative_model_G(state, action, node.belief_map, visited_cells, current_visited_cells, depth)

        # Tree descent: check if child node exists 
        if (action, observation) in node.children:
            child_node = node.children[(action, observation)]
        else:
            _, next_drone_pos = next_state
            new_belief_map = self.get_updated_belief_map_with_sensors(node.belief_map, next_drone_pos, observation, self.sensor_alpha, self.sensor_beta)
            
            child_node = POMCPNode(belief_map=new_belief_map, parent=node)
            node.children[(action, observation)] = child_node

            self.total_nodes_created += 1

        # Recursive call of simulate function for q-value calculation
        if terminal:
            future_reward = 0.0
        else:
            future_reward = self.simulate(next_state, child_node, depth + 1, visited_cells, current_visited_cells)

        q_value = reward + self.gamma * future_reward

        # Backpropagation: update N(b), N(b,a) and Q(b,a)
        node.total_node_visits += 1
        node.action_counts[action] += 1
        
        old_q = node.q_value_actions[action]        
        node.q_value_actions[action] = old_q + (q_value - old_q) / node.action_counts[action]       # Incremental update of average Q(b,a): Q_new = Q_old + (q_value - Q_old) / N(b,a)

        return q_value

    # Node expansion by initializing admissible actions
    def expand(self, node, state, partner_positions=None):
        
        _, drone_pos = state  # Extract current drone position from state
        
        # Extract only partner positions from dictionary {drone_id: position}
        partner_list = []
        if partner_positions is not None:
            partner_list = list(partner_positions.values())

        for action in MOVES_DELTA.keys():

            delta = MOVES_DELTA[action]
            next_pos = (drone_pos[0] + delta[0], drone_pos[1] + delta[1])

            # Check map boundaries
            if not (0 <= next_pos[0] < self.map_size and 0 <= next_pos[1] < self.map_size):
                continue

            # FORBID cells with obstacles
            if self.obstacle_map[next_pos[0], next_pos[1]] == 1:
                continue

            # FORBID cells occupied by partners (only at root, depth=0)
            if partner_list and next_pos in partner_list:
                continue

            
            node.action_counts[action] = 0         # N(b,a)
            node.q_value_actions[action] = 0.0     # Q(b,a)

    # Rollout based on real distance (BFS) between drone and sampled target
    def rollout(self, state):
        
        target_pos, drone_pos = state
        
        dist = self.dist_BFS.get((drone_pos, target_pos))
        
        # If the pair does not exist in the lookup table, the target is unreachable
        if dist is None:
            return 0.0  # Minimum reward for unreachable target
        
        # Formula for a decreasing reward with BFS distance
        score = (0.75 ** dist)
        return score

    # Black box simulator: state transition (drone movement), observation, reward
    def generative_model_G(self, state, action, belief_map, visited_cells, current_visited_cells, depth=0):
        
        target_pos, drone_pos = state 
        
        # Simulate drone movement
        delta = MOVES_DELTA[action]
        next_drone = (drone_pos[0] + delta[0], drone_pos[1] + delta[1])
        next_state = (target_pos, next_drone)

        # Simulate observation generation for single drone (false positive: alpha - false negative: beta)
        if (next_drone == target_pos):
            obs = 0 if np.random.rand() < self.sensor_beta else 1
        else:
            obs = 1 if np.random.rand() < self.sensor_alpha else 0
                     
        terminal = False        # Flag to indicate if target was found, to not continue tree descent 

        # Base reward calculation: R_base = R_target + reward_alpha * R_token 
        if (next_drone == target_pos) and obs == 1:
            r_target_reward = self.r_target          # Maximum reward for finding the target
            terminal = True     
        else:
            r_target_reward = 0.0          # No reward if target was not found 

        if next_drone not in visited_cells:
            r_token = belief_map[next_drone]        # Additional reward for exploring new cells, proportional to probability that target is in that cell according to belief map
        else:
            r_token = 0.0                           # No reward for already visited cells

        base_reward = r_target_reward + (self.reward_alpha * r_token)

        explorative_bonus = self.explorative_reward if next_drone not in current_visited_cells else 0.0

        visited_cells.add(next_drone)
        current_visited_cells.add(next_drone)

        total_reward = base_reward + explorative_bonus

        return next_state, obs, total_reward, terminal

    # Bayesian update of belief map using explicitly provided sensor parameters
    def get_updated_belief_map_with_sensors(self, current_belief, drone_pos, observation, alpha_sensor, beta_sensor):
        
        # Definition of Psi and Phi    
        if observation == 1:
            # Positive Detection
            Psi = 1.0 - beta_sensor  # True Positive
            Phi = alpha_sensor       # False Positive
        else:
            # Negative Detection
            Psi = beta_sensor        # False Negative
            Phi = 1.0 - alpha_sensor # True Negative

        # Calculate intermediate terms
        Omega = Psi - Phi
        p_st = current_belief[drone_pos]

        # Calculate normalization factor Z 
        Z = Phi + (Omega * p_st)

        # Numeric protection to avoid division by zero
        if Z < 1e-9:
            return current_belief 

        # Calculate new belief map: bayesian update formula
        new_belief_map = (current_belief.copy() * Phi) / Z 
        new_belief_map[drone_pos] = (Psi * p_st) / Z
        
        # Apply obstacles: zero probability in cells with obstacles
        new_belief_map = new_belief_map * (1 - self.obstacle_map)
        
        # Normalize new belief map to ensure probabilities sum to 1
        total = np.sum(new_belief_map)

        if total > 1e-9:  # Protection against zero sum
            new_belief_map /= total
        else:
            # Limiting case: if normalization fails initialize with uniform map
            free_cells_mask = (1 - self.obstacle_map)
            num_free_cells = np.sum(free_cells_mask)
            new_belief_map = free_cells_mask / num_free_cells

        return new_belief_map


    # Build a virtual belief map by simulating negative partner observations along future plans
    def build_virtual_belief_map(self, current_belief_map, my_pos, partner_positions, partner_plans):
        virtual_belief_map = current_belief_map.copy()

        if not partner_positions or not partner_plans:
            return virtual_belief_map       # If no partner information is available, return the original belief map

        # Keep only partners that are not excluded by proximity+priority rule.
        simulated_positions = {}
        for partner_id, partner_pos in partner_positions.items():
            dist_manhattan = abs(my_pos[0] - partner_pos[0]) + abs(my_pos[1] - partner_pos[1])
            if not (dist_manhattan <= 4 and partner_id > self.drone_id):
                simulated_positions[partner_id] = partner_pos

        if not simulated_positions:     # If all partners are excluded by proximity+priority rule, return the original belief map
            return virtual_belief_map

        # Apply negative observation update for current positions of partners
        for partner_id, current_pos in simulated_positions.items():
            virtual_belief_map = self.get_updated_belief_map_with_sensors(virtual_belief_map, current_pos, 0, self.sensor_alpha, self.sensor_beta)

        # Determine the maximum horizon for the decay of sensor parameters
        max_horizon = 0
        for partner_id, plan in partner_plans.items():
            if partner_id in simulated_positions and plan:
                max_horizon = max(max_horizon, len(plan))

        if max_horizon == 0:            # If no partner has a valid plan, return the original belief map
            return virtual_belief_map

        for step_idx in range(max_horizon):
            # t+1 uses real sensor parameters, future steps progressively degrade toward 0.5 at maximum horizon
            progress = 0.0 if max_horizon == 1 else step_idx / (max_horizon - 1)
            effective_alpha = self.sensor_alpha + (0.1 - self.sensor_alpha) * progress
            effective_beta = self.sensor_beta + (0.1 - self.sensor_beta) * progress

            cells_to_update = set()

            for partner_id, current_pos in list(simulated_positions.items()):
                plan = partner_plans.get(partner_id, [])
                if step_idx >= len(plan):
                    continue

                # Simulate the partner's next position based on their plan
                action = plan[step_idx]
                delta = MOVES_DELTA.get(action, (0, 0))
                next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])

                simulated_positions[partner_id] = next_pos
                cells_to_update.add(next_pos)

            # Apply negative observation update for all future simulated positions of partners (once per cell, for each step)
            for cell in cells_to_update:
                virtual_belief_map = self.get_updated_belief_map_with_sensors(virtual_belief_map, cell, 0, effective_alpha, effective_beta)

        return virtual_belief_map

    
    # Extract target position for POMCP
    def _sample_target_from_belief(self, belief_map):
        
        flat_probs = belief_map.flatten()
        
        # Protection: normalize if sum does not exactly equal 1 due to approximations (to avoid numeric issues)
        total = np.sum(flat_probs)
        if abs(total - 1.0) > 1e-6:  # Tolleranza numerica
            if total > 1e-9:
                flat_probs = flat_probs / total
            else:
                # Fallback: distribuzione uniforme
                flat_probs = np.ones_like(flat_probs) / flat_probs.size
        
        indices = np.arange(belief_map.size)
        sampled_idx = np.random.choice(indices, p=flat_probs)
        x, y = np.unravel_index(sampled_idx, belief_map.shape)
        
        return (x, y)

    # UCB function for action selection
    def _ucb_search(self, node):

        best_val = -float('inf') # Initialize to -inf to ensure any valid action surpasses it
        ucb_best_action = None 

        log_total_visits = math.log(node.total_node_visits) if node.total_node_visits > 0 else 0 
        infinite_actions = []

        for action in node.action_counts.keys():
            n_ba = node.action_counts[action]
            q_ba = node.q_value_actions[action]

            # If N(b,a) is zero, assign infinite UCT value to ensure this action is explored at least once
            if n_ba == 0:
                uct_val = float('inf')
                infinite_actions.append(action) # Save actions with infinite UCT in a list for subsequent random choice
            else:
                uct_val = q_ba + self.c * math.sqrt(log_total_visits / n_ba) # UCB formula

            if uct_val > best_val:
                best_val = uct_val
                ucb_best_action = action

        if infinite_actions:
            return random.choice(infinite_actions) # Random choice among actions with infinite UCT in the list

        return ucb_best_action

    # Selection of best action
    def _select_best_action(self, node):
        
        if not node.q_value_actions:
            return 'Stay'
        
        # Find action with highest Q-value
        best_action = max(node.q_value_actions.items(), key=lambda x: x[1])[0]
        
        return best_action

    # Extraction of future plans for all actions from root, following most promising moves
    def _extract_future_plans(self, root):

        future_plans = {}
        
        for root_action in root.action_counts.keys():
            plan = [root_action]  # Plan starts with root action
            
            # Find all children corresponding to this action (root level)
            matching_children = [(action_obs, child) for action_obs, child in root.children.items() if action_obs[0] == root_action]
            
            if not matching_children:
                # No child found: plan contains only root action
                future_plans[root_action] = [root_action]
                continue
            
            # Choose most visited child among those corresponding to the action
            most_visited_child = max(matching_children, key=lambda x: x[1].total_node_visits)[1]
            current_node = most_visited_child
            
            # Extract subsequent best actions up to maximum of 4 (for total plan of 5 actions)
            for _ in range(4):
    
                if not current_node.q_value_actions or current_node.total_node_visits == 0:
                    # No existing action from this node or never visited
                    break
                
                # Filter only actions actually explored at least once (N(b,a) > 0)
                visited_actions = {action: q_val for action, q_val in current_node.q_value_actions.items() if current_node.action_counts.get(action, 0) > 0}
                
                if not visited_actions:
                    break
                
                # Choose action with highest Q-value among filtered ones
                best_action_here = max(visited_actions.items(), key=lambda x: x[1])[0]
                
                plan.append(best_action_here)
                
                # Find all children corresponding to this chosen action
                matching_children_next = [(action_obs, child) for action_obs, child in current_node.children.items() if action_obs[0] == best_action_here]
                
                if not matching_children_next:
                    break
                
                # Advance to most visited child node among those with chosen action
                current_node = max(matching_children_next, key=lambda x: x[1].total_node_visits)[1]
            
            # Save plan with actually reached length (from 1 to 5 actions)
            future_plans[root_action] = plan
        
        return future_plans




# =============================================================================
# MULTIPROCESSING WORKER FUNCTION FOR POMCP
# =============================================================================

# Worker function for parallel POMCP (function for multiprocessing Pool)
def worker_pomcp_task(params, belief_map, my_pos, partner_positions, partner_plans, drone_id, obstacle_map, explored_cells):
    
    # Create POMCP solver instance with necessary parameters and data
    solver = POMCPSolver(
        max_time=params['max_time'],
        depth_limit=params['depth_limit'],
        discount_factor=params['discount_factor'],
        exploration_const=params['exploration_const'],
        sensor_alpha=params['alpha_sensor'],
        sensor_beta=params['beta_sensor'],
        reward_alpha=params['reward_alpha'],
        explorative_reward=params['explorative_reward'],
        map_size=params['map_size'],
        obstacle_map=obstacle_map,
        drone_id=drone_id,
        dist_BFS=params['dist_BFS'],
        r_target=params['r_target'],
        explored_cells=explored_cells
    )
    
    # Build virtual belief map by simulating negative partner observations along their future plans
    virtual_belief_map = solver.build_virtual_belief_map(belief_map, my_pos, partner_positions, partner_plans)

    # Execute POMCP with virtual preprocessed belief map
    best_action, future_plans = solver.search(virtual_belief_map, my_pos, partner_positions, partner_plans)
    
    return {
        'best_action': best_action,
        'depth': solver.max_depth_reached,
        'visits': solver.root.total_node_visits,
        'nodes_created': solver.total_nodes_created,
        'future_plans': future_plans
    }

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
# 4. DRONE AGENT 
# =============================================================================

class DroneAgent:
    
    def __init__(self, drone_id, start_pos, params, partner_positions=None):

        self.id = drone_id      # Drone id
        self.params = params    # Initial configuration parameters
        
        self.search_mode = 'TSP' if params['map_type'] == 1 else 'POMCP'    # Set search mode based on map type
        
        self.belief_map = initialize_belief_map(params) # Belief map initialization

        self.explored_cells = set()     # Set of physically visited cells during the mission
        
        self.obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else np.zeros((params['map_size'], params['map_size']), dtype=int) # Obstacle map initialization
        
        # Solver used only to update own belief_map (actual planning is executed in parallel in worker_pomcp_task function)
        self.solver_tool = POMCPSolver(
            max_time=params['max_time'],
            depth_limit=params['depth_limit'],
            discount_factor=params['discount_factor'],
            exploration_const=params['exploration_const'],
            sensor_alpha=params['alpha_sensor'],
            sensor_beta=params['beta_sensor'],
            reward_alpha=params['reward_alpha'],
            explorative_reward=params['explorative_reward'],
            map_size=params['map_size'],
            obstacle_map=self.obstacle_map,
            drone_id=self.id,
            dist_BFS=params['dist_BFS'],
            r_target=params['r_target'],
            explored_cells=self.explored_cells
        )

        self.planned_result = None      # Dictionary to store POMCP result (best_action, future_plans, depth, visits, nodes_created)

        self.pos = start_pos            # Current drone position
        self.final_action = None        # Contains final action to execute
        self.future_plans_buffer = {}   # Buffer for own future plans (including root action)
        self.observation = None         # Last observation received from real sensor
        self.positive_obs_count = 0     # Counter for positive observations received for TSP mode

        self.partner_positions = partner_positions if partner_positions is not None else {}     # Dictionary to store current partner positions received
        self.partner_final_actions = {}     # Dictionary to store intentions (best_action) received from partners
        self.partner_future_plans = {}      # Dictionary to store partner future plans received
        self.partner_observations = {}      # Dictionary to store observations received from partners
        
        self.discovered_traces = set()      # Set to track already discovered traces (by position) to avoid reprocessing
        
        if self.search_mode == 'TSP':
            self.tsp_plan = deque()         # Initialize tsp_plan if TSP mode is active
            self.tsp_solver = TSPSolver(
                map_size=params['map_size'],
                obstacle_map=self.obstacle_map,
                drone_id=self.id,
                start_pos=self.pos,
                darp_matrix=params.get('darp_assignment')
            )   
            full_plan = self.tsp_solver.generate_full_plan()        # Generate full plan and populate tsp_plan
            self.tsp_plan.extend(full_plan)



    # Method that simulates sending own movement intention to companion
    def send_intention(self):
        
        return {'id': self.id, 'pos': self.pos, 'best_action': self.planned_result['best_action']}


    # Method that simulates receiving partner's intention and stores it
    def receive_intention(self, drone_id, position, best_action):
        self.partner_final_actions[drone_id] = best_action
        self.partner_positions[drone_id] = position

    # Method to resolve conflicts locally using own information and received from partners, updating final action to execute
    def resolve_conflicts_local(self):
        
        all_drones_info = {self.id: (self.pos, self.final_action)}      # Create complete dictionary {drone_id: (position, action)} for all involved drones (self + partners)
        
        for partner_id, partner_action in self.partner_final_actions.items():
            partner_pos = self.partner_positions[partner_id]
            all_drones_info[partner_id] = (partner_pos, partner_action)
        
        # Drone movement: calculate future position of each drone based on own predicted action (best_action) and current position
        future_positions = {}
        for drone_id, (pos, action) in all_drones_info.items():
            delta = MOVES_DELTA[action]
            future_pos = (pos[0] + delta[0], pos[1] + delta[1])
            future_positions[drone_id] = future_pos
        
        # Identify conflicts: group drones by future position {future_pos: [drone_id1, drone_id2, ...]}
        pos_to_drones = {}
        for drone_id, future_pos in future_positions.items():
            if future_pos not in pos_to_drones:
                pos_to_drones[future_pos] = []
            pos_to_drones[future_pos].append(drone_id)
        
        # Risolvi conflitti: per ogni posizione futura con più droni, solo ID minore mantiene la mossa, gli altri sono forzati a Stay
        final_actions = {}
        for future_pos, drone_ids in pos_to_drones.items():
            if len(drone_ids) > 1: # Check conflict
                
                drone_ids_sorted = sorted(drone_ids) 
                winner_id = drone_ids_sorted[0]
                for drone_id in drone_ids:
                    if drone_id == winner_id:
                        final_actions[drone_id] = all_drones_info[drone_id][1]  # Drone with lower ID keeps original action
                    else:
                        final_actions[drone_id] = 'Stay'  # Others are forced to Stay
            else:
                
                drone_id = drone_ids[0]
                final_actions[drone_id] = all_drones_info[drone_id][1] # No conflict, keeps original action
        
        # Update own attributes with resolved final actions
        original_action = self.final_action
        self.final_action = final_actions[self.id]
        
        for partner_id in self.partner_final_actions.keys():
            self.partner_final_actions[partner_id] = final_actions[partner_id]
        

        if self.final_action != original_action:
            print(f"  [D{self.id}] Conflitto rilevato: {original_action} → {self.final_action} (Stay forzato)")     # Debug log for detected conflicts and modified actions


    # Method to execute movement by updating own physical position based on resolved final action
    def execute_move(self):
        
        d = MOVES_DELTA.get(self.final_action, (0, 0))
        self.pos = (self.pos[0] + d[0], self.pos[1] + d[1])


    # Simulate real drone sensor observation, with trace detection logic and target detection logic
    def get_real_observation(self, target_pos, traces):
        
        # Check if current position has a trace that hasn't been discovered yet
        trace_found = None
        for trace in traces:
            if trace['pos'] == self.pos and self.pos not in self.discovered_traces:
                trace_found = trace
                break
        
        if trace_found:
            # Trace detection logic (only False Negative, no False Positive)
            if np.random.rand() < self.params['beta_sensor']:
                self.observation = 0    # False Negative: trace is present but not detected
            else:
                self.observation = trace_found  # Trace is detected, save trace information in observation attribute
        else:
            # Target detection logic (or normal observation on already-discovered trace cell)
            if (self.pos == target_pos):
                obs = 0 if np.random.rand() < self.params['beta_sensor'] else 1
            else:
                obs = 1 if np.random.rand() < self.params['alpha_sensor'] else 0
            
            self.observation = obs


    # Method that simulates sending own observation and future plan to partners
    def send_observation_and_future_plan(self):
        
        executed_plan = self.future_plans_buffer.get(self.final_action, [])     # Get future plan for final action executed
        future_plan = executed_plan[1:] if len(executed_plan) > 1 else []       # Exclude first action (already executed) from communicated plan
        
        return {
            'id': self.id,
            'pos': self.pos,
            'observation': self.observation,
            'future_plan': future_plan
        }
    

    # Method that simulates receiving observation and future plan from partner, storing information in internal registers
    def receive_remote_observation(self, drone_id, position, observation, future_plan):
        
        # Save partner position, observation and future plan
        self.partner_positions[drone_id] = position
        self.partner_observations[drone_id] = observation
        self.partner_future_plans[drone_id] = future_plan
        

    # Method to apply trace distribution to belief map using Decentralized Data Fusion
    def apply_trace_distribution(self, trace_obs):
        
        trace_type = trace_obs['type']
        trace_pos = trace_obs['pos']
        trace_params = trace_obs['trace_params']
        map_size = self.params['map_size']
        
        # Create grid of coordinates and calculate deltas from trace position
        r, c = np.indices((map_size, map_size)) 
        delta_r = r - trace_pos[0]
        delta_c = c - trace_pos[1]

        # Generate distribution based on trace type
        if trace_type == 'von_mises':
            mu = trace_params['mu']             # Direction in radians
            kappa = trace_params['kappa']       # Concentration parameter
            
            # Calculate angles from trace position to each cell
            angles = np.arctan2(delta_r, delta_c)

            # Calculate von Mises distribution values for each cell
            diff = angles - mu
            trace_distribution = np.exp(kappa * np.cos(diff)) / (2 * np.pi * np.i0(kappa))

            # Set distribution value to 1 at the trace position cell (maximum likelihood)
            trace_distribution[trace_pos[0], trace_pos[1]] = 1.0
        
        elif trace_type == 'ring':
            radius = trace_params['radius']      # Radius of the ring
            variance = trace_params['variance']  # Variance of the ring
            
            # Calculate distance from trace position for each cell
            dist_matrix = np.sqrt(delta_r**2 + delta_c**2)

            # Calculate ring-shaped distribution values for each cell
            trace_distribution = np.exp(-((dist_matrix - radius)**2) / (2 * variance))
        
        elif trace_type == 'gaussian':
            sigma_x = trace_params['sigma_x']   # Standard deviation in x direction
            sigma_y = trace_params['sigma_y']   # Standard deviation in y direction
            
            # Create grid coordinates
            x, y = np.mgrid[0:map_size, 0:map_size]
            coord = np.dstack((x, y))
            
            # Covariance matrix
            cov_matrix = [[sigma_x**2, 0], [0, sigma_y**2]]
            
            # Create multivariate normal distribution
            rv = multivariate_normal(trace_pos, cov_matrix)
            trace_distribution = rv.pdf(coord)
        

        # Combines independent likelihood from trace with current belief
        fused_belief = self.belief_map * trace_distribution     # Hadamard product (element-wise multiplication)
        fused_belief = fused_belief * (1 - self.obstacle_map)   # Apply obstacles (zero probability in obstacle cells)
        total_prob = np.sum(fused_belief)       # Normalization 

        if total_prob > 1e-9:
            fused_belief /= total_prob
        else:
            # Fallback: keep previous belief unchanged (skip trace update)
            fused_belief = self.belief_map.copy()
        
        return fused_belief


    # Method to update belief map from all observations (own and partners), processing traces first then standard Bayesian updates
    def update_belief_from_all_obs(self):
        
        # Collect and unify all observations from this turn
        all_observations = []
        
        all_observations.append((self.pos, self.observation))   # Add own observation
        for partner_id, partner_obs in self.partner_observations.items():   # Add partners' observations
            partner_pos = self.partner_positions[partner_id]
            all_observations.append((partner_pos, partner_obs))
        
        # Process all observations: traces (if not already discovered) or standard Bayesian updates
        for pos, obs in all_observations:
            if isinstance(obs, dict) and 'type' in obs and 'pos' in obs:    # Check if observation is a trace
                if pos not in self.discovered_traces:       # Check if this trace was already discovered in a previous turn
                    self.belief_map = self.apply_trace_distribution(obs)
                    self.discovered_traces.add(pos)
                    print(f"  [D{self.id}] New trace discovered at {pos} of type '{obs['type']}'")
                    # Switch from TSP to POMCP mode when trace is detected
                    if self.search_mode == 'TSP':
                        self.search_mode = 'POMCP'
                        print(f"  [D{self.id}] Switching from TSP to POMCP mode due to trace detection")
                else:
                    print(f"  [D{self.id}] Trace at {pos} already processed, skipping")
            elif isinstance(obs, int) and obs in [0, 1]:    # Standard Bayesian update for non-trace observations
                self.belief_map = self.solver_tool.get_updated_belief_map_with_sensors(self.belief_map, pos, obs, self.solver_tool.sensor_alpha, self.solver_tool.sensor_beta)
                self.explored_cells.add(pos)
        
        # Clean buffers for next turn
        self.partner_observations.clear()

# =============================================================================
# 4. GRAPHIC FUNCTIONS 
# =============================================================================

# Draw grid, heatmap and percentages on background
# Draw grid, heatmap, DARP regions, and percentages on background
def draw_static_background(graphics_ctx, belief_map, drones=None):
    
    surface = graphics_ctx['background_surface']
    cell_size = graphics_ctx['CELL_SIZE']
    font_cell = graphics_ctx['font_cell']
    params = graphics_ctx['params']
    map_size = params['map_size']
    
    max_prob = belief_map.max()
    obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else None
    
    # Recuperiamo la matrice DARP se è stata calcolata (es. mappa uniforme)
    darp_matrix = params.get('darp_assignment', None)
    drone_start_positions = params.get('drone_positions', [])
    
    drone_colors = DARP_AREA_COLORS

    surface.fill(COLORS['WHITE'])

    is_pomcp_mode = False
    if drones is not None:
        is_pomcp_mode = any(getattr(drone, 'search_mode', None) == 'POMCP' for drone in drones)

    # Draw grid with coloring based on belief map, obstacles, and DARP
    for r in range(map_size):
        for c in range(map_size):
            x = c * cell_size
            y = r * cell_size
            prob = belief_map[r, c]

            if obstacle_map is not None and obstacle_map[r, c] == 1:
                color = COLORS['BLACK']
            else:
                
                if not is_pomcp_mode and darp_matrix is not None and (darp_matrix[r, c] != -1 or (r, c) in drone_start_positions):
                    if darp_matrix[r, c] != -1:
                        drone_idx = darp_matrix[r, c]
                    else:
                        drone_idx = drone_start_positions.index((r, c))
                        
                    # Prendi il colore corrispondente al drone
                    base_color = drone_colors[drone_idx % len(drone_colors)]
                    
                    if max_prob > 1e-9:
                        # Calcola l'intensità in base alla probabilità (Heatmap)
                        intensity = ((prob / max_prob) ** 0.4)
                        # Interpola il colore: base pastello (20%) + saturazione basata sulla prob (fino all'80%)
                        color = (
                            int(255 - (255 - base_color[0]) * (0.2 + 0.6 * intensity)),
                            int(255 - (255 - base_color[1]) * (0.2 + 0.6 * intensity)),
                            int(255 - (255 - base_color[2]) * (0.2 + 0.6 * intensity))
                        )
                    else:
                        # Colore pastello molto chiaro se probabilità è ~0
                        color = (
                            int(255 - (255 - base_color[0]) * 0.2),
                            int(255 - (255 - base_color[1]) * 0.2),
                            int(255 - (255 - base_color[2]) * 0.2)
                        )
                else:
                    # Logica originale (solo heatmap blu) se DARP non è usato o se in modalità POMCP
                    if max_prob > 1e-9:
                        color_val = int(255 * ((prob / max_prob) ** 0.4))
                        color = (255 - color_val, 255 - color_val, 255)
                    else:
                        color = (255, 255, 255)

            pygame.draw.rect(surface, color, (x, y, cell_size, cell_size))
            pygame.draw.rect(surface, COLORS['BLACK'], (x, y, cell_size, cell_size), 1)

            # Stampa la percentuale
            if obstacle_map is None or obstacle_map[r, c] == 0:
                text = font_cell.render(f"{prob * 100:.3f}%", True, COLORS['BLACK'])
                text_rect = text.get_rect(centerx=x + cell_size // 2, bottom=y + cell_size - max(2, cell_size // 20))
                surface.blit(text, text_rect)
                
# Draw dynamic elements: drones, target, traces and sidebar with statistics
def draw_elements(graphics_ctx, drones, target_pos, traces, stats):
    
    screen = graphics_ctx['screen']
    CELL_SIZE = graphics_ctx['CELL_SIZE']
    GRID_WIDTH = graphics_ctx['GRID_WIDTH']
    SIDEBAR_WIDTH = graphics_ctx['SIDEBAR_WIDTH']
    font_cell = graphics_ctx['font_cell']
    font_sidebar = graphics_ctx['font_sidebar']
    font_sidebar_fixed = graphics_ctx['font_sidebar_fixed']
    spacing = graphics_ctx['spacing']
    belief_map = drones[0].belief_map
    
    # Target
    tx, ty = target_pos
    target_rect = pygame.Rect(ty * CELL_SIZE, tx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.line(screen, COLORS['RED'], target_rect.topleft, target_rect.bottomright, 3)
    pygame.draw.line(screen, COLORS['RED'], target_rect.topright, target_rect.bottomleft, 3)

    # Traces
    for trace in traces:
        trace_r, trace_c = trace['pos']
        trace_square = pygame.Rect(trace_c * CELL_SIZE + CELL_SIZE // 4, trace_r * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(screen, COLORS['ORANGE'], trace_square, 0)  # Filled square
        
        center = (trace_c * CELL_SIZE + CELL_SIZE // 2, trace_r * CELL_SIZE + CELL_SIZE // 2)
        trace_label = font_cell.render("Tr", True, COLORS['WHITE'])
        screen.blit(trace_label, trace_label.get_rect(center=center))

    # Drones
    drone_colors = list(COLORS.values())[8:]
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

    # Statistics for each drone adapted based on number of drones
    y_offset = 10
    for drone in drones:
        color = drone_colors[(drone.id - 1) % len(drone_colors)]
        
        screen.blit(font_sidebar.render(f"=== Drone {drone.id} ===", True, color), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing

        screen.blit(font_sidebar.render(f"Sims: {stats['drones'][drone.id].get('visits', 0)}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing

        screen.blit(font_sidebar.render(f"Depth: {stats['drones'][drone.id].get('depth', 0)}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing

        screen.blit(font_sidebar.render(f"Best: {stats['drones'][drone.id].get('best', '-')}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing

        screen.blit(font_sidebar.render(f"Final: {stats['drones'][drone.id].get('final', '-')}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing

        if stats['drones'][drone.id].get('conflict', False):
            screen.blit(font_sidebar.render("⚠ Conflict!", True, COLORS['LIGHT_RED']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing

    # Fixed elements at bottom
    screen_height = screen.get_height()
    controls_y = screen_height - 26
    screen.blit(font_sidebar_fixed.render("SPAZIO: Avvia/Pausa  |  R: Riavvia  |  ESC: Esci", True, COLORS['BLACK']), (GRID_WIDTH + 10, controls_y))

    bar_y = controls_y - 33
    bar_width = SIDEBAR_WIDTH - 40
    max_prob = belief_map.max()
    pygame.draw.rect(screen, COLORS['WHITE'], (GRID_WIDTH + 20, bar_y, bar_width, 18))
    pygame.draw.rect(screen, COLORS['LIGHT_BLUE'], (GRID_WIDTH + 20, bar_y, bar_width * min(max_prob, 1.0), 18))
    
    pygame.draw.line(screen, COLORS['RED'], ((GRID_WIDTH + 20) + bar_width * 0.95, bar_y - 2), ((GRID_WIDTH + 20) + bar_width * 0.95, bar_y + 20), 2)
    
    max_prob_y = bar_y - 26
    screen.blit(font_sidebar_fixed.render(f"Max Prob: {max_prob * 100:.2f}%", True, COLORS['BLACK']), (GRID_WIDTH + 20, max_prob_y))

    text_thr = font_sidebar_fixed.render("Threshold: 95%", True, COLORS['RED'])
    screen.blit(text_thr, (GRID_WIDTH + SIDEBAR_WIDTH - 20 - text_thr.get_width(), max_prob_y))
    
    screen.blit(font_sidebar_fixed.render(f"Step: {stats['step']}", True, COLORS['BLACK']), (GRID_WIDTH + 20, max_prob_y - 16))


# Graphic system initialization
def init_graphics(params):
    
    pygame.init()

    map_size = params['map_size']
    display_info = pygame.display.Info()
    
    sidebar_w = 400
    cell_size = min((int(display_info.current_w * 0.90) - sidebar_w) // map_size, int(display_info.current_h * 0.90) // map_size)

    GRID_WIDTH = map_size * cell_size
    screen_w = min(GRID_WIDTH + sidebar_w, int(display_info.current_w * 0.90))
    screen_h = min(map_size * cell_size, int(display_info.current_h * 0.90))
    
    print(f"\n{'='*30}")
    print(f"Risoluzione: {display_info.current_w}x{display_info.current_h}")
    print(f"Mappa: {map_size}x{map_size}")
    print(f"Celle: {cell_size}px")
    print(f"Finestra: {screen_w}x{screen_h}px")
    print(f"{'='*30}\n")

    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Multi-Drone POMCP Decentralized")

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


# Disegna i percorsi TSP completi
def draw_tsp_paths(graphics_ctx, drones):
    """
    Independent graphic function to draw the calculated TSP paths.
    - Draws a solid thin line with directional arrows at the cell boundaries.
    - Turns the line/arrow red and double-directional if passing through a cell multiple times.
    - Uses transparent colors to keep underlying text readable.
    """
    screen = graphics_ctx['screen']
    CELL_SIZE = graphics_ctx['CELL_SIZE']
    
    # Crea una superficie temporanea con supporto alpha per la trasparenza
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    def draw_boundary_arrow(surface, color, start_pos, end_pos, width=1, is_double=False):
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Disegna una linea continua sottile
        pygame.draw.line(surface, color, (x1, y1), (x2, y2), width)
        
        # Metà esatta tra i due centri, sul confine tra le due celle
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = max(5, CELL_SIZE * 0.20)
        arrow_rad = 0.5  # Angolo apertura punta
        
        # Funzione interna per disegnare la punta (a forma di V)
        def draw_head(cx, cy, ang):
            p1 = (cx - arrow_len * math.cos(ang - arrow_rad), cy - arrow_len * math.sin(ang - arrow_rad))
            p2 = (cx - arrow_len * math.cos(ang + arrow_rad), cy - arrow_len * math.sin(ang + arrow_rad))
            pygame.draw.line(surface, color, (cx, cy), p1, width)
            pygame.draw.line(surface, color, (cx, cy), p2, width)
            
        # Punta primaria verso l'arrivo
        draw_head(mx, my, angle)
        
        if is_double:
            # Punta secondaria (verso la partenza) per il doppio senso
            draw_head(mx, my, angle + math.pi)

    from collections import Counter
    
    has_paths = False
    
    for drone in drones:
        if getattr(drone, 'search_mode', None) == 'TSP' and hasattr(drone, 'tsp_plan'):
            if not drone.tsp_plan:
                continue

            # Simula il piano di movimento futuro
            current_r, current_c = drone.pos
            path_cells = [(current_r, current_c)]
            
            for action in drone.tsp_plan:
                dr, dc = MOVES_DELTA.get(action, (0, 0))
                if action == 'Stay':
                    continue
                current_r += dr
                current_c += dc
                path_cells.append((current_r, current_c))
                
            if len(path_cells) < 2:
                continue
            
            has_paths = True
            
            # Conta le occorrenze di ciascuna cella nel percorso previsto
            cell_visits = Counter(path_cells)
            
            for i in range(len(path_cells) - 1):
                r1, c1 = path_cells[i]
                r2, c2 = path_cells[i+1]
                
                # Centri delle celle
                x1 = c1 * CELL_SIZE + CELL_SIZE // 2
                y1 = r1 * CELL_SIZE + CELL_SIZE // 2
                x2 = c2 * CELL_SIZE + CELL_SIZE // 2
                y2 = r2 * CELL_SIZE + CELL_SIZE // 2
                
                # Se la cella di partenza o di arrivo del segmento è passata più di una volta
                is_overlap = cell_visits[(r1, c1)] > 1 or cell_visits[(r2, c2)] > 1
                
                if is_overlap:
                    color = (255, 0, 0, 110)  # Rosso trasparente (alpha=110)
                    double_arrow = True
                else:
                    color = (0, 0, 0, 110)    # Nero trasparente (alpha=110)
                    double_arrow = False
                    
                # Spessore 1 per massima finezza, disegnato sull'overlay trasparente
                draw_boundary_arrow(overlay, color, (x1, y1), (x2, y2), width=1, is_double=double_arrow)
                
    # Applica l'overlay trasparente allo schermo reale solo se abbiamo disegnato qualcosa
    if has_paths:
        screen.blit(overlay, (0, 0))


# Complete frame rendering
def render_frame(graphics_ctx, drones, target_pos, traces, ui_stats):
    
    draw_static_background(graphics_ctx, drones[0].belief_map, drones)
    
    graphics_ctx['screen'].fill(COLORS['WHITE'])
    graphics_ctx['screen'].blit(graphics_ctx['background_surface'], (0, 0))
    
    draw_elements(graphics_ctx, drones, target_pos, traces, ui_stats)
    
    # CHIAMATA MOMENTANEA ALLA GRAFICA PER TSP
    draw_tsp_paths(graphics_ctx, drones)
    
    pygame.display.flip()


# =============================================================================
# MAIN LOOP
# =============================================================================

# Main function that executes multi-agent simulation with decentralized POMCP
def run_simulation(params):
    
    graphics_ctx = init_graphics(params)
    
    num_drones = params['num_drones']
    target_pos = params['target_pos']
    drone_positions_list = params['drone_positions']
    traces = params['traces']
    
    # Create drone parameters without environment secrets (target position and traces)
    drone_params = {k: v for k, v in params.items() if k not in ['target_pos', 'traces']}
    
    # DRONE AGENTS AND SIDEBAR STATISTICS INITIALIZATION
    drones = []
    for i in range(num_drones):

        # Calculate partner positions and pass them to drone constructor
        partner_pos = {j + 1: drone_positions_list[j] for j in range(num_drones) if j != i}
        drone = DroneAgent(i + 1, drone_positions_list[i], drone_params, partner_pos)
        drones.append(drone)
    
    # Sidebar statistics initialized
    ui_stats = {
        'step': 0,
        'drones': {drone.id: {
            'obs': '-',
            'depth': 0,
            'visits': 0,
            'nodes': 0,
            'best': '-',
            'final': '-',
            'conflict': False
        } for drone in drones}
    }

    # SETUP MULTIPROCESSING
    pool = multiprocessing.Pool(processes=num_drones)
    
    # Parameters for main simulation loop
    running = True
    auto_mode = False
    step_counter = 0
    move_interval_sec = 0.5
    last_step_time = 0.0

    # MAIN LOOP
    try:
        while running:
            
            if auto_mode and (time.monotonic() - last_step_time) >= move_interval_sec:
                last_step_time = time.monotonic()

                step_counter += 1
                print(f"\n--- STEP {step_counter} ---")


                # Check if all TSP plans are empty for each drones to switch to POMCP
                all_tsp_empty = True
                for drone in drones:
                    if hasattr(drone, 'tsp_plan') and len(drone.tsp_plan) > 0:
                        all_tsp_empty = False
                        break
                if all_tsp_empty:
                    for drone in drones:
                        if getattr(drone, 'search_mode', None) == 'TSP':
                            drone.search_mode = 'POMCP'
                            print(f"  [D{drone.id}] Switching from TSP to POMCP mode because all TSP plans are empty")


                # PARALLEL PLANNING (POMCP) or SYSTEMATIC EXPLORATION (TSP)
                tasks = []
                for drone in drones:
                    if drone.search_mode == 'TSP':
                        
                        # Management of positive obs in TSP mode
                        if drone.observation == 1:
                            drone.positive_obs_count += 1       # Increment of positive obs count to force Stay in next turns
                            drone.tsp_plan.appendleft('Stay')
                            
                        elif drone.observation == 0 and drone.positive_obs_count > 0:
                            drone.positive_obs_count -= 1       # Decrement of positive obs count to return to normal TSP moves
                            
                            if drone.positive_obs_count > 0:
                                drone.tsp_plan.appendleft('Stay')

                        # Action execution in TSP list
                        if drone.tsp_plan:
                            action = drone.tsp_plan.popleft()
                            drone.final_action = action
                        else:
                            drone.final_action = 'Stay'
                            
                        # DA AGGIUSTARE QUANDO FAREMO GRAFICA TSP
                        drone.planned_result = {
                            'best_action': drone.final_action,
                            'depth': 0,
                            'visits': 0,
                            'nodes_created': 0,
                            'future_plans': {}
                        }
                        drone.future_plans_buffer = {}

                    elif drone.search_mode == 'POMCP':
                        task = (drone.params, drone.belief_map.copy(), drone.pos, drone.partner_positions.copy(), drone.partner_future_plans.copy(), drone.id, drone.obstacle_map, drone.explored_cells.copy())
                        tasks.append(task)
                
                if tasks:
                    # Multiprocessing pool to execute POMCPs in parallel for each drone, collecting results in a list
                    results = pool.starmap(worker_pomcp_task, tasks)
                    
                    # Update each drone with its own POMCP results
                    for i, drone in enumerate(drones):
                        drone.planned_result = results[i]
                        drone.final_action = results[i]['best_action']
                        drone.future_plans_buffer = results[i]['future_plans']
                    
                    # Print future plans for all drones
                    print(f"\n=== FUTURE PLANS FOR ALL DRONES ===")
                    for drone in drones:
                        print(f"\nDrone {drone.id} (pos {drone.pos}):")
                        for action, plan in drone.future_plans_buffer.items():
                            print(f"  If action '{action}': {' -> '.join(plan)}")

                    # Each drone sends its intention (position and predicted action) to other drones
                    intention_packets = [drone.send_intention() for drone in drones]
                    
                    # Each drone receives intentions from other drones and updates its attributes (partner_final_actions and partner_positions)
                    for drone in drones:
                        for pkt in intention_packets:
                            if pkt['id'] != drone.id:
                                drone.receive_intention(pkt['id'], pkt['pos'], pkt['best_action'])
                    
                    # CONFLICT RESOLUTION
                    for drone in drones:
                        drone.resolve_conflicts_local()

                # MOVEMENT
                for drone in drones:
                    drone.execute_move()

                # PERCEPTION (for all drones: both TSP and POMCP)
                for drone in drones:
                    drone.get_real_observation(target_pos, traces)
                
                observation_packets = [drone.send_observation_and_future_plan() for drone in drones]
                
                # OBSERVATION COMMUNICATION (for all drones: both TSP and POMCP)
                for drone in drones:
                    for pkt in observation_packets:
                        if pkt['id'] != drone.id:
                            drone.receive_remote_observation(pkt['id'], pkt['pos'], pkt['observation'], pkt['future_plan'])
                
                # BELIEF UPDATE (for all drones: both TSP and POMCP)
                for drone in drones:
                    drone.update_belief_from_all_obs()

                # Update statistics for each drone to show in sidebar
                ui_stats['step'] = step_counter
                for drone in drones:
                    result = drone.planned_result
                    had_conflict = (drone.final_action != result['best_action'])
                    ui_stats['drones'][drone.id].update({
                        'obs': drone.observation,
                        'depth': result['depth'],
                        'visits': result['visits'],
                        'nodes': result['nodes_created'],
                        'best': result['best_action'],
                        'final': drone.final_action,
                        'conflict': had_conflict
                    })

                # Simulation termination condition (target found with probability >= 95%, which is the threshold set)
                if drones[0].belief_map.max() >= 0.95:
                    print("\n TARGET TROVATO! (probabilità > 95%)")
                    auto_mode = False
                
            render_frame(graphics_ctx, drones, target_pos, traces, ui_stats)
            graphics_ctx['clock'].tick(30)
            
            # User input handling 
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
                            last_step_time = time.monotonic() - move_interval_sec
                            print("\n✓ Modalità AUTO POMCP attivata")
                        else:
                            print("\n✓ Modalità AUTO disattivata")
            
    finally:
        pool.close()    # Close multiprocessing pool
        pool.join()     # Wait for all pool processes to terminate


def main():
    while True:
        params = get_user_parameters()
        result = run_simulation(params)
        if result == "quit":
            print("Simulazione terminata.")
            pygame.quit()
            break
        elif result == "restart":
            print("Riavvio...")
            pygame.quit()
            pygame.init()
            continue

if __name__ == "__main__":
    main()
