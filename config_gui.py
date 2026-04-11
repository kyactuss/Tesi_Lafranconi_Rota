import pygame
import math
import numpy as np
from collections import deque
from scipy.ndimage import label, distance_transform_edt

# =============================================================================
# 1. PARAMETERS CONFIGURATION E COSTANTI GLOBALI
# =============================================================================
DEFAULT_CONFIG = {
    'map_size': 20,
    'alpha_sensor': 0.01,
    'beta_sensor': 0.01,
    'max_time': 2.5,
    'depth_limit': 1000,
    'discount_factor': 0.95,
    'exploration_const': math.sqrt(2),
    'reward_alpha': 3,                  #DEFAULT: 1
    'explorative_reward': 0.005,       #DEFAULT: 0.0025
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

# =============================================================================
# 2. FUNZIONI DI SUPPORTO ALLA CONFIGURAZIONE
# =============================================================================
def initialize_obstacle_map(params):
    map_size = params['map_size']
    obstacle_map = np.zeros((map_size, map_size), dtype=int)
    for obs_pos in params.get('obstacles', []):
        r, c = obs_pos
        obstacle_map[r, c] = 1
    return obstacle_map

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


# =============================================================================
# 3. INTERFACCIA DI CONFIGURAZIONE UTENTE
# =============================================================================
def get_user_parameters(scenario_idx=1):
    
    print(f"\n=== SEARCH MISSION CONFIGURATION (SCENARIO {scenario_idx}) ===")
    
    map_size = DEFAULT_CONFIG['map_size']
    alpha_sensor = DEFAULT_CONFIG['alpha_sensor']
    beta_sensor = DEFAULT_CONFIG['beta_sensor']

    print(f"Parameters: Map Size={map_size}x{map_size}, Alpha={alpha_sensor}, Beta={beta_sensor}")
    
    pygame.init()
    
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    
    margin = 25  
    info_height = 55  
    bottom_margin = 35
    
    max_usable_width = screen_width * 0.9
    max_usable_height = screen_height * 0.9
    
    cell_size_from_width = int((max_usable_width - 2 * margin) / map_size)
    cell_size_from_height = int((max_usable_height - info_height - margin - bottom_margin) / map_size)
    cell_size = min(cell_size_from_width, cell_size_from_height)
    
    window_width = map_size * cell_size + 2 * margin
    window_height = map_size * cell_size + info_height + margin + bottom_margin
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"Mission Configuration Scenario {scenario_idx} - Click to select")
    
    font_title = pygame.font.Font(None, 28)
    font_info = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 18)
    font_selected = pygame.font.Font(None, 20)
    font_cell = pygame.font.Font(None, max(1, int(cell_size*0.5)))
    
    def draw_grid(selected_positions, phase_name, phase_color, context_items=None):
        screen.fill(COLORS['WHITE'])
        
        if context_items is None:
            context_items = {}
        
        if 'peaks' in context_items and context_items['peaks']:
            if isinstance(context_items['peaks'][0], dict):
                peaks_positions = [peak['mean'] for peak in context_items['peaks']]
            else:
                peaks_positions = context_items['peaks']
        else:
            peaks_positions = []
        
        if 'traces' in context_items and context_items['traces']:
            traces_positions = [trace['pos'] for trace in context_items['traces']]
        else:
            traces_positions = []
        
        title_text = font_title.render(f"PHASE: {phase_name}", True, phase_color)
        info_text = font_info.render("Click on grid to select cells", True, COLORS['BLACK'])
        help_text = font_small.render("Press ENTER to confirm and proceed to next phase", True, COLORS['RED'])

        screen.blit(title_text, (margin, 10))
        screen.blit(info_text, (margin, 35))
        screen.blit(help_text, (margin, 60))
        
        grid_offset_y = margin + info_height
        for r in range(map_size):
            for c in range(map_size):
                x = margin + c * cell_size
                y = grid_offset_y + r * cell_size
                
                cell_color = COLORS['LIGHT_GRAY']
                label_text = f"{r},{c}"
                text_color = COLORS['BLACK']
                
                if 'drones' in context_items and (r, c) in context_items['drones']:
                    cell_color = COLORS['BLUE']
                    label_text = "D"
                    text_color = COLORS['WHITE']
                elif 'target' in context_items and (r, c) == context_items['target']:
                    cell_color = COLORS['RED']
                    label_text = "T"
                    text_color = COLORS['WHITE']
                elif (r, c) in peaks_positions:
                    cell_color = COLORS['PURPLE']
                    label_text = "G"
                    text_color = COLORS['WHITE']
                elif (r, c) in traces_positions:
                    cell_color = COLORS['ORANGE']
                    label_text = "Tr"
                    text_color = COLORS['WHITE']
                elif 'obstacles' in context_items and (r, c) in context_items['obstacles']:
                    cell_color = COLORS['BLACK']
                    label_text = "X"
                    text_color = COLORS['WHITE']
                elif (r, c) in selected_positions:
                    cell_color = phase_color
                    label_text = f"{r},{c}"
                    text_color = COLORS['WHITE']
                
                pygame.draw.rect(screen, cell_color, (x, y, cell_size, cell_size))
                pygame.draw.rect(screen, COLORS['GRAY'], (x, y, cell_size, cell_size), 1)
                
                coord_text = font_cell.render(label_text, True, text_color)
                text_rect = coord_text.get_rect(center=(x + cell_size//2, y + cell_size//2))
                screen.blit(coord_text, text_rect)
        
        count_text = font_selected.render(f"Selected: {len(selected_positions)}", True, COLORS['BLACK'])
        screen.blit(count_text, (margin, window_height - bottom_margin + 10))
        pygame.display.flip()
    
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
                            selected.remove(cell)
                        else:
                            if validate_cell:
                                if cell in context_items.get('drones', []) or cell == context_items.get('target'):
                                    print(f"  Cannot place obstacle on drone/target!")
                                    continue
                            
                            if allow_multiple:
                                selected.append(cell)
                            else:
                                selected = [cell]
                
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
        'obstacles': dictionary_context['obstacles'],
        'scenario_idx': scenario_idx  # Salviamo l'indice dello scenario nei parametri per usarlo nello screenshot
    }
    
    obstacle_map = initialize_obstacle_map(params_dict)
    dist_BFS = precompute_BFS_distances(map_size, obstacle_map)
    params_dict['dist_BFS'] = dist_BFS      
    
    if params_dict['map_type'] == 1: 
        print("\n[DARP] Starting DARP algorithm...")
        assignment_matrix = darp_partitioning(params_dict)
        params_dict['darp_assignment'] = assignment_matrix
        print("[DARP] Division completed successfully.")

    return params_dict