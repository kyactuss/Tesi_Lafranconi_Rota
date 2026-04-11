import random
import math
import numpy as np
from collections import deque

from config_gui import (
    DEFAULT_CONFIG, 
    initialize_obstacle_map, 
    precompute_BFS_distances
)

def generate_random_parameters(scenario_idx, map_size=20):
    print(f"\n[+] Procedural generation of Scenario {scenario_idx} in progress...")
    
    # 1. Drones: Fixed to 4 for now
    num_drones = 4
    
    params = {
        'map_size': map_size,
        'alpha_sensor': DEFAULT_CONFIG['alpha_sensor'],
        'beta_sensor': DEFAULT_CONFIG['beta_sensor'],
        'max_time': DEFAULT_CONFIG['max_time'],
        'depth_limit': DEFAULT_CONFIG['depth_limit'],
        'discount_factor': DEFAULT_CONFIG['discount_factor'],
        'exploration_const': DEFAULT_CONFIG['exploration_const'],
        'reward_alpha': DEFAULT_CONFIG['reward_alpha'],
        'explorative_reward': DEFAULT_CONFIG['explorative_reward'],
        'r_target': DEFAULT_CONFIG['r_target'],
        'num_drones': num_drones,
        'scenario_idx': scenario_idx
    }

    # Available corners for drone starting positions
    corners = [
        (0, 0), 
        (0, map_size - 1), 
        (map_size - 1, 0), 
        (map_size - 1, map_size - 1)
    ]
    
    # =========================================================================
    # GENERATE VALID MAP (With BFS connectivity test)
    # =========================================================================
    while True:
        # A. Drone Placement (Exactly 1 drone per corner)
        params['drone_positions'] = corners.copy()
        
        # B. Target Placement (at least distance 3 from all drones)
        while True:
            tr = random.randint(0, map_size - 1)
            tc = random.randint(0, map_size - 1)
            target_pos = (tr, tc)
            
            # Chebyshev distance
            min_dist = min(max(abs(tr - dr), abs(tc - dc)) for dr, dc in params['drone_positions'])
            
            if min_dist >= 3 and target_pos not in params['drone_positions']:
                params['target_pos'] = target_pos
                break
                
        protected_cells = set(params['drone_positions'] + [params['target_pos']])
        
        # C. Maze-like Obstacle Generation (10% - 35%)
        total_cells = map_size * map_size
        target_obs_count = random.randint(int(total_cells * 0.10), int(total_cells * 0.35))
        obstacles = set()
        
        while len(obstacles) < target_obs_count:
            r, c = random.randint(0, map_size - 1), random.randint(0, map_size - 1)
            
            # Choose shape type to create corridors, dead ends, or small blocks
            shape_type = random.random()
            
            if shape_type < 0.25:
                # 25% chance: Standard Block (adds noise)
                w, h = random.randint(1, 3), random.randint(1, 3)
                for i in range(w):
                    for j in range(h):
                        curr_r, curr_c = r + i, c + j
                        if 0 <= curr_r < map_size and 0 <= curr_c < map_size:
                            if (curr_r, curr_c) not in protected_cells:
                                obstacles.add((curr_r, curr_c))
                                
            elif shape_type < 0.60:
                # 35% chance: Long Wall / Corridor
                length = random.randint(3, 7)
                dr, dc = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
                for i in range(length):
                    curr_r, curr_c = r + dr*i, c + dc*i
                    if 0 <= curr_r < map_size and 0 <= curr_c < map_size:
                        if (curr_r, curr_c) not in protected_cells:
                            obstacles.add((curr_r, curr_c))
                            
            else:
                # 40% chance: L-shape or U-shape (creates dead ends / cul-de-sacs)
                length1 = random.randint(3, 6)
                length2 = random.randint(3, 6)
                dr1, dc1 = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
                
                # Perpendicular direction
                dr2, dc2 = dc1, dr1 
                if random.random() < 0.5: dr2, dc2 = -dr2, -dc2
                
                # Draw first arm
                curr_r, curr_c = r, c
                for i in range(length1):
                    nr, nc = curr_r + dr1*i, curr_c + dc1*i
                    if 0 <= nr < map_size and 0 <= nc < map_size and (nr, nc) not in protected_cells:
                        obstacles.add((nr, nc))
                
                # Draw second arm from the end of the first
                curr_r, curr_c = curr_r + dr1*(length1-1), curr_c + dc1*(length1-1)
                for i in range(length2):
                    nr, nc = curr_r + dr2*i, curr_c + dc2*i
                    if 0 <= nr < map_size and 0 <= nc < map_size and (nr, nc) not in protected_cells:
                        obstacles.add((nr, nc))

        params['obstacles'] = list(obstacles)
        
        # D. Full Connectivity Test (BFS)
        obs_map = initialize_obstacle_map(params)
        free_cells_count = total_cells - len(obstacles)
        
        start_node = params['drone_positions'][0]
        queue = deque([start_node])
        visited = {start_node}
        
        while queue:
            curr_r, curr_c = queue.popleft()
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < map_size and 0 <= nc < map_size:
                    if obs_map[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        # If BFS reached all free cells, the map is perfectly playable
        if len(visited) == free_cells_count:
            params['dist_BFS'] = precompute_BFS_distances(map_size, obs_map)
            break 
        else:
            # Map is broken (contains inaccessible rooms). Discard and retry.
            pass

    # =========================================================================
    # MAP TYPE, PEAKS, AND TRACES
    # =========================================================================
    
    # Always Multi-Gaussian (2)
    params['map_type'] = 2
    
    all_coords = [(r, c) for r in range(map_size) for c in range(map_size)]
    free_coords = [pos for pos in all_coords if pos not in params['obstacles']]
    
    # --- PEAKS (Multi-Gaussian) ---
    params['peaks'] = []
    num_peaks = random.randint(1, 5)
    
    # 1. Close centers (Distance <= 10 from target)
    close_centers = [pos for pos in free_coords if max(abs(pos[0] - target_pos[0]), abs(pos[1] - target_pos[1])) <= 7]
    
    # 2. Far centers (Traps: Distance > 10 from target)
    far_centers = [pos for pos in free_coords if max(abs(pos[0] - target_pos[0]), abs(pos[1] - target_pos[1])) > 10]
    
    # Fallback just in case the map is too small or heavily obstructed
    if not close_centers: close_centers = free_coords
    if not far_centers: far_centers = free_coords
        
    # 1 guaranteed peak relatively close to target
    peak_centers = random.sample(close_centers, 1)
    
    # Remaining peaks act as probability traps further away
    if num_peaks > 1:
        actual_far_peaks = min(num_peaks - 1, len(far_centers))
        peak_centers.extend(random.sample(far_centers, actual_far_peaks))
    
    for center in peak_centers:
        params['peaks'].append({
            'mean': center,
            # Covariance constrained between 2.0 and 5.0
            'cov': [random.uniform(2.0, 5.0), random.uniform(2.0, 5.0)]
        })

    # --- TRACES ---
    # Traces completely removed as requested
    params['traces'] = []

    print(f"✓ Scenario {scenario_idx} ready!")
    print(f"  - Drones: {num_drones} (4 corners)")
    print(f"  - Map Type: Multi-Gaussian")
    print(f"  - Obstacles: {len(params['obstacles'])} cells (Maze-like)")
    print(f"  - Peaks: {len(params['peaks'])} (1 close, {len(peak_centers)-1} traps)")
    print(f"  - Traces: None")
    
    return params