import elkai
from collections import deque
import math
from scipy.ndimage import label, distance_transform_edt
import time
import numpy as np
from scipy.stats import multivariate_normal
import pygame

# =============================================================================
# 1. PARAMETERS CONFIGURATION 
# =============================================================================
DEFAULT_CONFIG = {
    'map_size': 20,
    'alpha_sensor': 0.01,
    'beta_sensor': 0.01
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
        
        # Extract only the positions of the peaks, not the full dict
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
                label_text = f"{r},{c}"
                text_color = COLORS['BLACK']
                
                # Check if it's an already placed drone
                if 'drones' in context_items and (r, c) in context_items['drones']:
                    cell_color = COLORS['BLUE']
                    label_text = "D"
                    text_color = COLORS['WHITE']
                # Check if it's the already placed target
                elif 'target' in context_items and (r, c) == context_items['target']:
                    cell_color = COLORS['RED']
                    label_text = "T"
                    text_color = COLORS['WHITE']
                # Check if it's an already placed Gaussian peak
                elif (r, c) in peaks_positions:
                    cell_color = COLORS['PURPLE']
                    label_text = "G"
                    text_color = COLORS['WHITE']
                # Check if it's an already placed trace
                elif (r, c) in traces_positions:
                    cell_color = COLORS['ORANGE']
                    label_text = "Tr"
                    text_color = COLORS['WHITE']
                # Check if it's an already placed obstacle
                elif 'obstacles' in context_items and (r, c) in context_items['obstacles']:
                    cell_color = COLORS['BLACK']
                    label_text = "X"
                    text_color = COLORS['WHITE']
                # Check if the cell is selected in the current phase
                elif (r, c) in selected_positions:
                    cell_color = phase_color
                    label_text = f"{r},{c}"
                    text_color = COLORS['WHITE']
                
                # Draw the cell
                pygame.draw.rect(screen, cell_color, (x, y, cell_size, cell_size))
                pygame.draw.rect(screen, COLORS['GRAY'], (x, y, cell_size, cell_size), 1)
                
                # Show coordinates or label at the center of the cell
                coord_text = font_cell.render(label_text, True, text_color)
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
    
    # Function to select cells interactively for each phase
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
                                selected.append(cell)
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
        screen.blit(text, (margin, 80 + i * 40))
    
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
        
        if trace_type_input == '1':
            while True:
                try:
                    mu_deg = float(input("  Enter direction μ (in degrees): "))
                    kappa = float(input("  Enter concentration κ: "))
                    if kappa >= 0:
                        break
                    print("  Concentration must be non-negative.")
                except ValueError:
                    print("  Invalid format. Please enter valid numbers.")

            mu = math.radians(mu_deg)
            trace_dict = {
                'pos': coord,
                'type': 'von_mises',
                'trace_params': {'mu': mu, 'kappa': kappa}
            }
        
        elif trace_type_input == '2':
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
        
        elif trace_type_input == '3':
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
    params_dict['dist_BFS'] = dist_BFS
    
    # Call DARP partitioning if uniform map is selected
    if params_dict['map_type'] == 1: 
        print("\n[DARP] Starting DARP algorithm...")
        assignment_matrix = darp_partitioning(params_dict)
        params_dict['darp_assignment'] = assignment_matrix
        print("[DARP] Division completed successfully.")

    return params_dict


# Function to generate boolean obstacle map (1 = obstacle, 0 = free) 
def initialize_obstacle_map(params):
    map_size = params['map_size']
    obstacle_map = np.zeros((map_size, map_size), dtype=int)
    for obs_pos in params.get('obstacles', []):
        r, c = obs_pos
        obstacle_map[r, c] = 1
    return obstacle_map

# Function to initialize the belief map
def initialize_belief_map(params):
    map_size = params['map_size']
    map_type = params['map_type']
    peaks = params['peaks']
    
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

# Function to perform DARP partitioning
def darp_partitioning(params, max_iter=80000, variate_weight=0.01, random_level=0.0001, limit_cells_diff=2, use_importance=True):
    map_size = params['map_size']
    num_drones = params['num_drones']
    drone_positions = params['drone_positions']
    obstacle_map = initialize_obstacle_map(params) 
    
    tot_obstacles = np.sum(obstacle_map) 
    tot_cells = map_size * map_size 
    free_cells = tot_cells - num_drones - tot_obstacles
    term_thr = 1 if free_cells % num_drones != 0 else 0

    list_mat_D = np.zeros((num_drones, map_size, map_size))
    for r in range(num_drones):
        start_pos = drone_positions[r]
        for i in range(map_size):
            for j in range(map_size):
                if obstacle_map[i, j] == 1 or (i, j) in drone_positions:
                    list_mat_D[r, i, j] = float('inf')
                else:
                    dist = params['dist_BFS'].get((start_pos, (i, j)))
                    list_mat_D[r, i, j] = dist if dist is not None else float('inf')
                    
    max_valid_dist = np.max(list_mat_D[list_mat_D != float('inf')]) 
    list_mat_D[list_mat_D == float('inf')] = max_valid_dist * 2
    
    cells_importance = np.zeros((num_drones, map_size, map_size))
    max_importance = np.zeros(num_drones)
    min_importance = np.full(num_drones, float('inf'))
    
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

    list_mat_D_copy = np.copy(list_mat_D)
    mat_A = np.zeros((map_size, map_size), dtype=int)
    list_uavs_cells = np.zeros(num_drones, dtype=int)
    list_connected_regions = np.zeros(num_drones, dtype=bool)
    
    success = False 
    
    while term_thr <= limit_cells_diff and not success:
        down_thres = (tot_cells - term_thr * (num_drones - 1)) / (tot_cells * num_drones)
        upper_thres = (tot_cells + term_thr) / (tot_cells * num_drones)
        
        success = True
        iter_count = 0
        
        while iter_count <= max_iter:
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
                        mat_A[i, j] = -1

            for r, pos in enumerate(drone_positions):
                list_personal_assignment[r, pos[0], pos[1]] = 1
                
            list_mat_C = []
            plainErrors = np.zeros(num_drones) 
            divFairError = np.zeros(num_drones) 
            
            for r in range(num_drones):
                normalized_mat_C = np.ones((map_size, map_size))
                list_connected_regions[r] = True
                
                adj_cells = np.array([[0,1,0],
                                      [1,1,1],
                                      [0,1,0]]) 
                labeled_array, num_islands = label(list_personal_assignment[r], structure=adj_cells)
                
                if num_islands > 1:
                    list_connected_regions[r] = False
                    start_label = labeled_array[drone_positions[r][0], drone_positions[r][1]]
                    Ri_reg = (labeled_array == start_label).astype(int)
                    Qi_reg = ((list_personal_assignment[r] == 1) & (labeled_array != start_label)).astype(int)
                    
                    dist_to_uav = distance_transform_edt(1 - Ri_reg)
                    dist_to_island = distance_transform_edt(1 - Qi_reg)
                    
                    mat_C = dist_to_uav - dist_to_island
                    max_v, min_v = np.max(mat_C), np.min(mat_C)
                    if max_v > min_v:
                        normalized_mat_C = (mat_C - min_v) * ((2 * variate_weight) / (max_v - min_v)) + (1 - variate_weight)
                
                list_mat_C.append(normalized_mat_C)
                
                plainErrors[r] = list_uavs_cells[r] / free_cells
                if plainErrors[r] < down_thres:
                    divFairError[r] = down_thres - plainErrors[r]
                elif plainErrors[r] > upper_thres:
                    divFairError[r] = upper_thres - plainErrors[r]
            
            max_cells_ass = np.max(list_uavs_cells)
            min_cells_ass = np.min(list_uavs_cells)
            if (max_cells_ass - min_cells_ass) <= term_thr and np.all(list_connected_regions):
                break
                 
            total_neg_perc = np.sum(np.abs(divFairError[divFairError < 0]))
            total_neg_plain_errors = np.sum(plainErrors[divFairError < 0]) 
            
            for r in range(num_drones):
                coeff_m = 1.0
                if total_neg_plain_errors != 0.0:
                    if divFairError[r] < 0.0:
                        coeff_m = 1.0 + (plainErrors[r] / total_neg_plain_errors) * (total_neg_perc / 2.0)
                    else:
                        coeff_m = 1.0 - (plainErrors[r] / total_neg_plain_errors) * (total_neg_perc / 2.0)
                
                criterionMatrix = np.copy(cells_importance[r])
                if use_importance:
                    diff_imp = max_importance[r] - min_importance[r]
                    if divFairError[r] < 0:
                        criterionMatrix = (cells_importance[r] - min_importance[r]) * ((coeff_m - 1) / diff_imp) + 1
                    else:
                        criterionMatrix = (cells_importance[r] - min_importance[r]) * ((1 - coeff_m) / diff_imp) + coeff_m
                else:
                    criterionMatrix.fill(coeff_m)
                    
                RM = 2.0 * random_level * np.random.rand(map_size, map_size) + 1.0 - random_level
                list_mat_D_copy[r] = list_mat_D_copy[r] * criterionMatrix * RM * list_mat_C[r]

            iter_count += 1
            
        if iter_count >= max_iter:
            max_iter //= 2
            success = False
            term_thr += 1

    return mat_A

# BFS algorithm to precompute distances between cells
def precompute_BFS_distances(map_size, obstacle_map):
    dist_BFS = {}
    directions = [delta for action, delta in MOVES_DELTA.items() if action != 'Stay']
    
    for start_r in range(map_size):
        for start_c in range(map_size):
            start_pos = (start_r, start_c)
            
            if obstacle_map[start_r, start_c] == 1:
                continue
            
            queue = deque([(start_pos, 0)])
            visited = {start_pos}
            
            while queue:
                current_pos, dist = queue.popleft()
                dist_BFS[(start_pos, current_pos)] = dist
                
                for dr, dc in directions:
                    next_r = current_pos[0] + dr
                    next_c = current_pos[1] + dc
                    next_pos = (next_r, next_c)
                    
                    if not (0 <= next_r < map_size and 0 <= next_c < map_size):
                        continue
                    if obstacle_map[next_r, next_c] == 1:
                        continue
                    if next_pos in visited:
                        continue
                    
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))
    
    return dist_BFS

# =============================================================================
# 2. AUCTION LOGIC & TSP
# =============================================================================

class AUCTIONSolver:
    def __init__(self, map_size, obstacle_map, dist_BFS):
        self.map_size = map_size
        self.obstacle_map = obstacle_map
        self.dist_lookup = dist_BFS

    def calculate_utility_map(self, p_map, drone_pos):
        """Calcola la mappa di utilità U = P/(d+1) per un drone usando distanze reali (BFS)"""
        grid_w, grid_h = p_map.shape
        utility_map = np.full((grid_w, grid_h), -np.inf) 
        
        for r in range(grid_w):
            for c in range(grid_h):
                if self.obstacle_map[r, c] == 1:
                    continue
                cell_pos = (r, c)
                dist = self.dist_lookup.get((tuple(drone_pos), cell_pos))
                if dist is not None:
                    utility_map[r, c] = p_map[r, c] / (dist + 1.0)
        
        return utility_map

    def create_wish_list(self, utility_map):
        """Crea lista dei desideri ordinata dalla migliore alla peggiore"""
        grid_w, grid_h = utility_map.shape
        wish_list = []
        for r in range(grid_w):
            for c in range(grid_h):
                if utility_map[r, c] > -np.inf:
                    wish_list.append((utility_map[r, c], (r, c)))
        
        wish_list.sort(reverse=True, key=lambda x: x[0])
        return wish_list

    def get_next_step_bfs(self, current_pos, target_pos):
        """Calcola il prossimo passo ottimale verso il target usando il percorso BFS."""
        current_pos = tuple(current_pos)
        target_pos = tuple(target_pos)
        
        if current_pos == target_pos:
            return current_pos
        
        if (current_pos, target_pos) not in self.dist_lookup:
            return current_pos
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        best_next_pos = current_pos
        best_distance = float('inf')
        
        for dr, dc in directions:
            next_r = current_pos[0] + dr
            next_c = current_pos[1] + dc
            next_pos = (next_r, next_c)
            
            if not (0 <= next_r < self.map_size and 0 <= next_c < self.map_size):
                continue
            
            if self.obstacle_map[next_r, next_c] == 1:
                continue
            
            dist_to_target = self.dist_lookup.get((next_pos, target_pos))
            if dist_to_target is None:
                continue
            
            if dist_to_target < best_distance:
                best_distance = dist_to_target
                best_next_pos = next_pos
        
        return best_next_pos

    def calculate_AUCTION(self, my_id, my_pos, partner_positions, p_map):
        """
        Sistema di asta deterministico per assegnare celle target ai droni.
        Eseguito localmente restituisce l'azione risolta per il singolo drone.
        """
        # Creiamo un dizionario di tutti i droni per risolvere l'asta in modo globale-locale deterministico
        all_drones = {my_id: my_pos}
        all_drones.update(partner_positions)
        
        num_drones = len(all_drones)
        drone_ids = sorted(list(all_drones.keys()))
        
        drone_data = []
        for d_id in drone_ids:
            pos = all_drones[d_id]
            utility_map = self.calculate_utility_map(p_map, pos)
            wish_list = self.create_wish_list(utility_map)
            drone_data.append({
                'id': d_id,
                'pos': pos,
                'wish_list': wish_list,
                'wish_index': 0,
                'assigned_cell': None,
                'assigned': False
            })
        
        cell_bids = {}
        max_iterations = self.map_size * self.map_size * num_drones
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            any_change = False
            
            for drone in drone_data:
                if drone['assigned']:
                    continue
                
                if drone['wish_index'] >= len(drone['wish_list']):
                    drone['assigned_cell'] = drone['pos']
                    drone['assigned'] = True
                    continue
                
                utility, desired_cell = drone['wish_list'][drone['wish_index']]
                
                is_occupied_by_other = any(
                    d['pos'] == desired_cell and d['id'] != drone['id'] 
                    for d in drone_data
                )
                
                if is_occupied_by_other:
                    drone['wish_index'] += 1
                    any_change = True
                    continue
                
                if desired_cell not in cell_bids:
                    cell_bids[desired_cell] = {
                        'drone_id': drone['id'],
                        'utility': utility
                    }
                else:
                    current_bid = cell_bids[desired_cell]
                    if utility > current_bid['utility']:
                        for d in drone_data:
                            if d['id'] == current_bid['drone_id']:
                                d['wish_index'] += 1
                                d['assigned'] = False
                                any_change = True
                                break
                        cell_bids[desired_cell] = {
                            'drone_id': drone['id'],
                            'utility': utility
                        }
                    else:
                        drone['wish_index'] += 1
                        any_change = True
            
            for cell, bid in cell_bids.items():
                drone_id = bid['drone_id']
                for d in drone_data:
                    if d['id'] == drone_id:
                        d['assigned_cell'] = cell
                        d['assigned'] = True
                        break
                        
            if not any_change:
                break
                
        # Previeni swap simultanei
        for i in range(num_drones):
            for j in range(i+1, num_drones):
                drone_i = drone_data[i]
                drone_j = drone_data[j]
                if (drone_i['assigned_cell'] == drone_j['pos'] and 
                    drone_j['assigned_cell'] == drone_i['pos']):
                    drone_j['assigned_cell'] = drone_j['pos']
                    
        # Calcola le mosse effettive per tutti
        final_moves = {}
        for drone in drone_data:
            target_cell = drone['assigned_cell']
            current_pos = drone['pos']
            next_pos = self.get_next_step_bfs(current_pos, target_cell)
            final_moves[drone['id']] = next_pos
            
        # Risolve collisioni
        for i in range(num_drones):
            for j in range(i+1, num_drones):
                id_i = drone_data[i]['id']
                id_j = drone_data[j]['id']
                if final_moves[id_i] == final_moves[id_j]:
                    final_moves[id_j] = all_drones[id_j] # Stay
                    
        # Traduce in azione per "me stesso"
        my_next_pos = final_moves[my_id]
        final_action = 'Stay'
        for action, delta in MOVES_DELTA.items():
            if (my_pos[0] + delta[0], my_pos[1] + delta[1]) == my_next_pos:
                final_action = action
                break
                
        return final_action


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
                    
        if self.start_pos not in free_cells:
            free_cells.insert(0, self.start_pos)
            local_obstacle_map[self.start_pos[0], self.start_pos[1]] = 0
            
        num_free_cells = len(free_cells)

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
# 3. DRONE AGENT 
# =============================================================================

class DroneAgent:
    
    def __init__(self, drone_id, start_pos, params, partner_positions=None):

        self.id = drone_id      # Drone id
        self.params = params    # Initial configuration parameters
        
        self.search_mode = 'TSP' if params['map_type'] == 1 else 'AUCTION'    # Set search mode based on map type
        self.belief_map = initialize_belief_map(params) # Belief map initialization

        self.explored_cells = set()     # Set of physically visited cells during the mission
        
        self.obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else np.zeros((params['map_size'], params['map_size']), dtype=int)
        
        # Solver AUCTION
        self.solver_tool = AUCTIONSolver(
            map_size=params['map_size'],
            obstacle_map=self.obstacle_map,
            dist_BFS=params['dist_BFS']
        )

        self.pos = start_pos            # Current drone position
        self.final_action = None        # Contains final action to execute
        self.observation = None         # Last observation received from real sensor
        self.positive_obs_count = 0     # Counter for positive observations received for TSP mode

        self.partner_positions = partner_positions if partner_positions is not None else {}     # Dictionary to store current partner positions received
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


    def get_updated_belief_map(self, current_belief, drone_pos, observation):
        """Bayesian update logic included natively in DroneAgent."""
        if observation == 1:
            Psi = 1.0 - self.params['beta_sensor']  # True Positive
            Phi = self.params['alpha_sensor']       # False Positive
        else:
            Psi = self.params['beta_sensor']        # False Negative
            Phi = 1.0 - self.params['alpha_sensor'] # True Negative

        Omega = Psi - Phi
        p_st = current_belief[drone_pos]

        Z = Phi + (Omega * p_st)

        if Z < 1e-9:
            return current_belief 

        new_belief_map = (current_belief.copy() * Phi) / Z 
        new_belief_map[drone_pos] = (Psi * p_st) / Z
        
        new_belief_map = new_belief_map * (1 - self.obstacle_map)
        
        total = np.sum(new_belief_map)

        if total > 1e-9:  
            new_belief_map /= total
        else:
            free_cells_mask = (1 - self.obstacle_map)
            num_free_cells = np.sum(free_cells_mask)
            new_belief_map = free_cells_mask / num_free_cells

        return new_belief_map


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


    # Method that simulates sending own observation to partners
    def send_observation(self):
        return {
            'id': self.id,
            'pos': self.pos,
            'observation': self.observation
        }
    

    # Method that simulates receiving observation from partner, storing information in internal registers
    def receive_remote_observation(self, drone_id, position, observation):
        self.partner_positions[drone_id] = position
        self.partner_observations[drone_id] = observation
        

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
            
            angles = np.arctan2(delta_r, delta_c)
            diff = angles - mu
            trace_distribution = np.exp(kappa * np.cos(diff)) / (2 * np.pi * np.i0(kappa))
            trace_distribution[trace_pos[0], trace_pos[1]] = 1.0
        
        elif trace_type == 'ring':
            radius = trace_params['radius']      # Radius of the ring
            variance = trace_params['variance']  # Variance of the ring
            
            dist_matrix = np.sqrt(delta_r**2 + delta_c**2)
            trace_distribution = np.exp(-((dist_matrix - radius)**2) / (2 * variance))
        
        elif trace_type == 'gaussian':
            sigma_x = trace_params['sigma_x']   
            sigma_y = trace_params['sigma_y']   
            
            x, y = np.mgrid[0:map_size, 0:map_size]
            coord = np.dstack((x, y))
            
            cov_matrix = [[sigma_x**2, 0], [0, sigma_y**2]]
            rv = multivariate_normal(trace_pos, cov_matrix)
            trace_distribution = rv.pdf(coord)
        

        # Combines independent likelihood from trace with current belief
        fused_belief = self.belief_map * trace_distribution     
        fused_belief = fused_belief * (1 - self.obstacle_map)   
        total_prob = np.sum(fused_belief)       

        if total_prob > 1e-9:
            fused_belief /= total_prob
        else:
            fused_belief = self.belief_map.copy()
        
        return fused_belief


    # Method to update belief map from all observations (own and partners), processing traces first then standard Bayesian updates
    def update_belief_from_all_obs(self):
        
        all_observations = []
        all_observations.append((self.pos, self.observation))   
        for partner_id, partner_obs in self.partner_observations.items():   
            partner_pos = self.partner_positions[partner_id]
            all_observations.append((partner_pos, partner_obs))
        
        for pos, obs in all_observations:
            if isinstance(obs, dict) and 'type' in obs and 'pos' in obs:    
                if pos not in self.discovered_traces:       
                    self.belief_map = self.apply_trace_distribution(obs)
                    self.discovered_traces.add(pos)
                    print(f"  [D{self.id}] New trace discovered at {pos} of type '{obs['type']}'")
                    # Switch from TSP to AUCTION mode when trace is detected
                    if self.search_mode == 'TSP':
                        self.search_mode = 'AUCTION'
                        print(f"  [D{self.id}] Switching from TSP to AUCTION mode due to trace detection")
                else:
                    print(f"  [D{self.id}] Trace at {pos} already processed, skipping")
            elif isinstance(obs, int) and obs in [0, 1]:    
                self.belief_map = self.get_updated_belief_map(self.belief_map, pos, obs)
                self.explored_cells.add(pos)
        
        # Clean buffers for next turn
        self.partner_observations.clear()


# =============================================================================
# 4. GRAPHIC FUNCTIONS 
# =============================================================================

# Draw grid, heatmap, DARP regions, and percentages on background
def draw_static_background(graphics_ctx, belief_map, drones=None):
    
    surface = graphics_ctx['background_surface']
    cell_size = graphics_ctx['CELL_SIZE']
    font_cell = graphics_ctx['font_cell']
    params = graphics_ctx['params']
    map_size = params['map_size']
    
    max_prob = belief_map.max()
    obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else None
    
    darp_matrix = params.get('darp_assignment', None)
    drone_start_positions = params.get('drone_positions', [])
    
    drone_colors = DARP_AREA_COLORS

    surface.fill(COLORS['WHITE'])

    is_auction_mode = False
    if drones is not None:
        is_auction_mode = any(getattr(drone, 'search_mode', None) == 'AUCTION' for drone in drones)

    # Draw grid with coloring based on belief map, obstacles, and DARP
    for r in range(map_size):
        for c in range(map_size):
            x = c * cell_size
            y = r * cell_size
            prob = belief_map[r, c]

            if obstacle_map is not None and obstacle_map[r, c] == 1:
                color = COLORS['BLACK']
            else:
                if not is_auction_mode and darp_matrix is not None and (darp_matrix[r, c] != -1 or (r, c) in drone_start_positions):
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
                
# Draw dynamic elements: drones, target, traces and sidebar with statistics
def draw_elements(graphics_ctx, drones, target_pos, traces):
    
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
        pygame.draw.rect(screen, COLORS['ORANGE'], trace_square, 0)  
        
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

        screen.blit(font_sidebar.render(f"Mode: {drone.search_mode}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing

        screen.blit(font_sidebar.render(f"Action: {drone.final_action}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing
        
        obs_text = "Trace" if isinstance(drone.observation, dict) else str(drone.observation)
        screen.blit(font_sidebar.render(f"Last Obs: {obs_text}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
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
    pygame.display.set_caption("Multi-Drone AUCTION Decentralized")

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
    screen = graphics_ctx['screen']
    CELL_SIZE = graphics_ctx['CELL_SIZE']
    
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    def draw_boundary_arrow(surface, color, start_pos, end_pos, width=1, is_double=False):
        x1, y1 = start_pos
        x2, y2 = end_pos
        
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
        
        if is_double:
            draw_head(mx, my, angle + math.pi)

    from collections import Counter
    
    has_paths = False
    
    for drone in drones:
        if getattr(drone, 'search_mode', None) == 'TSP' and hasattr(drone, 'tsp_plan'):
            if not drone.tsp_plan:
                continue

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
            
            cell_visits = Counter(path_cells)
            
            for i in range(len(path_cells) - 1):
                r1, c1 = path_cells[i]
                r2, c2 = path_cells[i+1]
                
                x1 = c1 * CELL_SIZE + CELL_SIZE // 2
                y1 = r1 * CELL_SIZE + CELL_SIZE // 2
                x2 = c2 * CELL_SIZE + CELL_SIZE // 2
                y2 = r2 * CELL_SIZE + CELL_SIZE // 2
                
                is_overlap = cell_visits[(r1, c1)] > 1 or cell_visits[(r2, c2)] > 1
                
                if is_overlap:
                    color = (255, 0, 0, 110)  
                    double_arrow = True
                else:
                    color = (0, 0, 0, 110)    
                    double_arrow = False
                    
                draw_boundary_arrow(overlay, color, (x1, y1), (x2, y2), width=1, is_double=double_arrow)
                
    if has_paths:
        screen.blit(overlay, (0, 0))


# Complete frame rendering
def render_frame(graphics_ctx, drones, target_pos, traces):
    
    draw_static_background(graphics_ctx, drones[0].belief_map, drones)
    
    graphics_ctx['screen'].fill(COLORS['WHITE'])
    graphics_ctx['screen'].blit(graphics_ctx['background_surface'], (0, 0))
    
    draw_elements(graphics_ctx, drones, target_pos, traces)
    
    draw_tsp_paths(graphics_ctx, drones)
    
    pygame.display.flip()


# =============================================================================
# MAIN LOOP
# =============================================================================

def run_simulation(params):
    
    graphics_ctx = init_graphics(params)
    
    num_drones = params['num_drones']
    target_pos = params['target_pos']
    drone_positions_list = params['drone_positions']
    traces = params['traces']
    
    drone_params = {k: v for k, v in params.items() if k not in ['target_pos', 'traces']}
    
    # DRONE AGENTS INITIALIZATION
    drones = []
    for i in range(num_drones):
        partner_pos = {j + 1: drone_positions_list[j] for j in range(num_drones) if j != i}
        drone = DroneAgent(i + 1, drone_positions_list[i], drone_params, partner_pos)
        drones.append(drone)
    
    running = True
    auto_mode = False
    step_counter = 0
    move_interval_sec = 1
    last_step_time = 0.0

    # MAIN LOOP
    while running:
        
        if auto_mode and (time.monotonic() - last_step_time) >= move_interval_sec:
            last_step_time = time.monotonic()

            step_counter += 1
            print(f"\n--- STEP {step_counter} ---")

            # Check if all TSP plans are empty for each drones to switch to AUCTION
            all_tsp_empty = True
            for drone in drones:
                if hasattr(drone, 'tsp_plan') and len(drone.tsp_plan) > 0:
                    all_tsp_empty = False
                    break
            if all_tsp_empty:
                for drone in drones:
                    if getattr(drone, 'search_mode', None) == 'TSP':
                        drone.search_mode = 'AUCTION'
                        print(f"  [D{drone.id}] Switching from TSP to AUCTION mode because all TSP plans are empty")

            # PLANNING (AUCTION or TSP)
            for drone in drones:
                if drone.search_mode == 'TSP':
                    
                    if drone.observation == 1:
                        drone.positive_obs_count += 1       
                        drone.tsp_plan.appendleft('Stay')
                        
                    elif drone.observation == 0 and drone.positive_obs_count > 0:
                        drone.positive_obs_count -= 1       
                        
                        if drone.positive_obs_count > 0:
                            drone.tsp_plan.appendleft('Stay')

                    if drone.tsp_plan:
                        action = drone.tsp_plan.popleft()
                        drone.final_action = action
                    else:
                        drone.final_action = 'Stay'

                elif drone.search_mode == 'AUCTION':
                    drone.final_action = drone.solver_tool.calculate_AUCTION(drone.id, drone.pos, drone.partner_positions, drone.belief_map)
            
            # MOVEMENT
            for drone in drones:
                drone.execute_move()

            # PERCEPTION
            for drone in drones:
                drone.get_real_observation(target_pos, traces)
            
            observation_packets = [drone.send_observation() for drone in drones]
            
            # OBSERVATION COMMUNICATION
            for drone in drones:
                for pkt in observation_packets:
                    if pkt['id'] != drone.id:
                        drone.receive_remote_observation(pkt['id'], pkt['pos'], pkt['observation'])
            
            # BELIEF UPDATE
            for drone in drones:
                drone.update_belief_from_all_obs()

            if drones[0].belief_map.max() >= 0.95:
                print("\n TARGET TROVATO! (probabilità > 95%)")
                auto_mode = False
            
        render_frame(graphics_ctx, drones, target_pos, traces)
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
                        print("\n✓ Modalità AUTO AUCTION attivata")
                    else:
                        print("\n✓ Modalità AUTO disattivata")


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
