import elkai
from collections import deque
import math
from scipy.ndimage import label, distance_transform_edt
import random
import time
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import entropy  
import pygame
import os

# =============================================================================
# 1. PARAMETERS CONFIGURATION & COSTANTI GLOBALI
# =============================================================================
DEFAULT_CONFIG = {
    'map_size': 20,
    'alpha_sensor': 0.01,
    'beta_sensor': 0.01,
}

MOVES_DELTA = {
    'N': (-1, 0),
    'S': (1, 0),
    'W': (0, -1),
    'E': (0, 1),
    'Stay': (0, 0)
}

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
# FUNZIONI DI SUPPORTO AMBIENTE
# =============================================================================

def initialize_obstacle_map(params):
    map_size = params['map_size']
    obstacle_map = np.zeros((map_size, map_size), dtype=int)
    if 'obstacles' in params:
        for obs_pos in params['obstacles']:
            r, c = obs_pos
            obstacle_map[r, c] = 1
    return obstacle_map

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
# GREEDY SOLVER E TSP SOLVER
# =============================================================================

class GREEDYSolver:
    def __init__(self, map_size, obstacle_map, drone_id, dist_BFS):
        self.map_size = map_size
        self.obstacle_map = obstacle_map
        self.drone_id = drone_id
        self.dist_BFS = dist_BFS

    def calculate_greedy(self, p_map, drone_pos, partner_positions):
        all_positions = {self.drone_id: drone_pos}
        all_positions.update(partner_positions)
        sorted_ids = sorted(all_positions.keys())
        claimed_next_positions = set()
        
        def get_best_move(d_id, d_pos):
            current_others = {pos for pid, pos in all_positions.items() if pid != d_id}
            blocked_cells = current_others.union(claimed_next_positions)
            candidates = []
            
            for action, delta in MOVES_DELTA.items():
                nr, nc = d_pos[0] + delta[0], d_pos[1] + delta[1]
                next_pos = (nr, nc)
                if not (0 <= nr < self.map_size and 0 <= nc < self.map_size):
                    continue
                if self.obstacle_map[nr, nc] == 1:
                    continue
                if action != 'Stay' and next_pos in blocked_cells:
                    continue
                candidates.append((action, p_map[nr, nc], next_pos))
                
            if not candidates:
                return 'Stay', d_pos
                
            best_action, max_prob, best_next_pos = max(candidates, key=lambda item: item[1])
            
            if max_prob < 0.0001:
                flat_index = np.argmax(p_map)
                target_pos = (flat_index // self.map_size, flat_index % self.map_size)
                if d_pos == target_pos:
                    return 'Stay', d_pos
                    
                best_distance = float('inf')
                fallback_action = 'Stay'
                fallback_next_pos = d_pos
                for action, prob, next_pos in candidates:
                    if action == 'Stay':
                        continue
                    dist_to_target = self.dist_BFS.get((next_pos, target_pos))
                    if dist_to_target is not None and dist_to_target < best_distance:
                        best_distance = dist_to_target
                        fallback_action = action
                        fallback_next_pos = next_pos
                if best_distance != float('inf'):
                    return fallback_action, fallback_next_pos
            return best_action, best_next_pos

        my_final_action = 'Stay'
        for pid in sorted_ids:
            p_pos = all_positions[pid]
            action, next_pos = get_best_move(pid, p_pos)
            if pid == self.drone_id:
                my_final_action = action
                break
            else:
                claimed_next_positions.add(next_pos)
        return my_final_action

class TSPSolver:
    def __init__(self, map_size, obstacle_map, drone_id, start_pos, darp_matrix):
        self.map_size = map_size
        self.obstacle_map = obstacle_map
        self.drone_id = drone_id
        self.start_pos = start_pos
        self.darp_matrix = darp_matrix
    
    def generate_full_plan(self):
        local_obstacle_map = np.copy(self.obstacle_map)
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
            
        try:
            tour_indices = elkai.solve_int_matrix(elk_matrix)
        except Exception as e:
            print(f"  [D{self.drone_id}] Error in elkai solver: {e}")
            return []
            
        start_idx = free_cells.index(self.start_pos)
        if start_idx in tour_indices:
            idx_in_tour = tour_indices.index(start_idx)
            ordered_tour = tour_indices[idx_in_tour:] + tour_indices[:idx_in_tour]
        else:
            ordered_tour = tour_indices
            
        ordered_tour.append(ordered_tour[0])
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
                    break
                actions.append(best_action)
                current_pos = best_next
        return actions

# =============================================================================
# 4. DRONE AGENT 
# =============================================================================

class DroneAgent:
    def __init__(self, drone_id, start_pos, params, partner_positions=None):
        self.id = drone_id
        self.params = params
        self.search_mode = 'TSP' if params['map_type'] == 1 else 'GREEDY'
        self.belief_map = initialize_belief_map(params)
        self.explored_cells = set()
        self.obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else np.zeros((params['map_size'], params['map_size']), dtype=int)
        self.greedy_solver = GREEDYSolver(params['map_size'], self.obstacle_map, self.id, params['dist_BFS'])
        
        self.pos = start_pos
        self.final_action = None
        self.observation = None
        self.positive_obs_count = 0

        self.partner_positions = partner_positions if partner_positions is not None else {}
        self.partner_observations = {}
        self.discovered_traces = set()
        
        if self.search_mode == 'TSP':
            self.tsp_plan = deque()
            self.tsp_solver = TSPSolver(
                map_size=params['map_size'],
                obstacle_map=self.obstacle_map,
                drone_id=self.id,
                start_pos=self.pos,
                darp_matrix=params.get('darp_assignment')
            )   
            full_plan = self.tsp_solver.generate_full_plan()
            self.tsp_plan.extend(full_plan)

    def execute_move(self):
        d = MOVES_DELTA.get(self.final_action, (0, 0))
        self.pos = (self.pos[0] + d[0], self.pos[1] + d[1])

    def get_real_observation(self, target_pos, traces):
        trace_found = None
        for trace in traces:
            if trace['pos'] == self.pos and self.pos not in self.discovered_traces:
                trace_found = trace
                break
        
        if trace_found:
            if np.random.rand() < self.params['beta_sensor']:
                self.observation = 0
            else:
                self.observation = trace_found
        else:
            if (self.pos == target_pos):
                obs = 0 if np.random.rand() < self.params['beta_sensor'] else 1
            else:
                obs = 1 if np.random.rand() < self.params['alpha_sensor'] else 0
            self.observation = obs

    def send_observation(self):
        return {'id': self.id, 'pos': self.pos, 'observation': self.observation}

    def receive_remote_observation(self, drone_id, position, observation):
        self.partner_positions[drone_id] = position
        self.partner_observations[drone_id] = observation

    def apply_trace_distribution(self, trace_obs):
        trace_type = trace_obs['type']
        trace_pos = trace_obs['pos']
        trace_params = trace_obs['trace_params']
        map_size = self.params['map_size']
        
        r, c = np.indices((map_size, map_size))
        delta_r = r - trace_pos[0]
        delta_c = c - trace_pos[1]

        if trace_type == 'von_mises':
            mu = trace_params['mu']
            kappa = trace_params['kappa']
            angles = np.arctan2(delta_r, delta_c)
            diff = angles - mu
            trace_distribution = np.exp(kappa * np.cos(diff)) / (2 * np.pi * np.i0(kappa))
            trace_distribution[trace_pos[0], trace_pos[1]] = 1.0
        elif trace_type == 'ring':
            radius = trace_params['radius']
            variance = trace_params['variance']
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
        
        fused_belief = self.belief_map * trace_distribution
        fused_belief = fused_belief * (1 - self.obstacle_map)
        total_prob = np.sum(fused_belief)

        if total_prob > 1e-9:
            fused_belief /= total_prob
        else:
            fused_belief = self.belief_map.copy()
        return fused_belief

    def get_updated_belief_map(self, current_belief, drone_pos, observation):
        if observation == 1:
            Psi = 1.0 - self.params['beta_sensor']
            Phi = self.params['alpha_sensor']
        else:
            Psi = self.params['beta_sensor']
            Phi = 1.0 - self.params['alpha_sensor']

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
                    if self.search_mode == 'TSP':
                        self.search_mode = 'GREEDY'
                        print(f"  [D{self.id}] Switching from TSP to GREEDY mode due to trace detection")
            elif isinstance(obs, int) and obs in [0, 1]:
                self.belief_map = self.get_updated_belief_map(self.belief_map, pos, obs)
                self.explored_cells.add(pos)
        self.partner_observations.clear()

# =============================================================================
# 4. GRAPHIC FUNCTIONS 
# =============================================================================

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

    is_greedy_mode = False
    if drones is not None:
        is_greedy_mode = any(getattr(drone, 'search_mode', None) == 'GREEDY' for drone in drones)

    for r in range(map_size):
        for c in range(map_size):
            x = c * cell_size
            y = r * cell_size
            prob = belief_map[r, c]

            if obstacle_map is not None and obstacle_map[r, c] == 1:
                color = COLORS['BLACK']
            else:
                if not is_greedy_mode and darp_matrix is not None and (darp_matrix[r, c] != -1 or (r, c) in drone_start_positions):
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

def draw_elements(graphics_ctx, drones, target_pos, traces, step_counter):
    screen = graphics_ctx['screen']
    CELL_SIZE = graphics_ctx['CELL_SIZE']
    GRID_WIDTH = graphics_ctx['GRID_WIDTH']
    SIDEBAR_WIDTH = graphics_ctx['SIDEBAR_WIDTH']
    font_cell = graphics_ctx['font_cell']
    font_sidebar = graphics_ctx['font_sidebar']
    font_sidebar_fixed = graphics_ctx['font_sidebar_fixed']
    spacing = graphics_ctx['spacing']
    belief_map = drones[0].belief_map
    
    tx, ty = target_pos
    target_rect = pygame.Rect(ty * CELL_SIZE, tx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.line(screen, COLORS['RED'], target_rect.topleft, target_rect.bottomright, 3)
    pygame.draw.line(screen, COLORS['RED'], target_rect.topright, target_rect.bottomleft, 3)

    for trace in traces:
        trace_r, trace_c = trace['pos']
        trace_square = pygame.Rect(trace_c * CELL_SIZE + CELL_SIZE // 4, trace_r * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(screen, COLORS['ORANGE'], trace_square, 0)
        
        center = (trace_c * CELL_SIZE + CELL_SIZE // 2, trace_r * CELL_SIZE + CELL_SIZE // 2)
        trace_label = font_cell.render("Tr", True, COLORS['WHITE'])
        screen.blit(trace_label, trace_label.get_rect(center=center))

    drone_colors = list(COLORS.values())[8:]
    for drone in drones:
        dr, dc = drone.pos
        center = (dc * CELL_SIZE + CELL_SIZE // 2, dr * CELL_SIZE + CELL_SIZE // 2)
        color = drone_colors[(drone.id - 1) % len(drone_colors)]
        pygame.draw.circle(screen, color, center, CELL_SIZE // 3, 4)
        
        id_text = font_cell.render(str(drone.id), True, color)
        screen.blit(id_text, id_text.get_rect(center=center))

    sidebar_rect = pygame.Rect(GRID_WIDTH, 0, SIDEBAR_WIDTH, screen.get_height())
    pygame.draw.rect(screen, COLORS['GRAY'], sidebar_rect)

    y_offset = 10
    for drone in drones:
        color = drone_colors[(drone.id - 1) % len(drone_colors)]
        screen.blit(font_sidebar.render(f"=== Drone {drone.id} ===", True, color), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing
        obs_val = "Tr" if isinstance(drone.observation, dict) else drone.observation
        screen.blit(font_sidebar.render(f"Action: {drone.final_action} | Obs: {obs_val}", True, COLORS['BLACK']), (GRID_WIDTH + 5, y_offset))
        y_offset += spacing * 1.5

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
    screen.blit(font_sidebar_fixed.render(f"Step: {step_counter}", True, COLORS['BLACK']), (GRID_WIDTH + 20, max_prob_y - 16))

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
    pygame.display.set_caption("Multi-Drone Decentralized - GREEDY")

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

def render_frame(graphics_ctx, drones, target_pos, traces, step_counter):
    draw_static_background(graphics_ctx, drones[0].belief_map, drones)
    graphics_ctx['screen'].fill(COLORS['WHITE'])
    graphics_ctx['screen'].blit(graphics_ctx['background_surface'], (0, 0))
    draw_elements(graphics_ctx, drones, target_pos, traces, step_counter)
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
    
    drones = []
    for i in range(num_drones):
        partner_pos = {j + 1: drone_positions_list[j] for j in range(num_drones) if j != i}
        drone = DroneAgent(i + 1, drone_positions_list[i], drone_params, partner_pos)
        drones.append(drone)

    running = True
    auto_mode = True           # Avvio automatico forzato
    step_counter = 0
    move_interval_sec = 0.0    # Simulazione rapida
    last_step_time = 0.0
    
    entropy_history = []       # Vettore per storicizzare l'entropia

    while running:
        if auto_mode and (time.monotonic() - last_step_time) >= move_interval_sec:
            last_step_time = time.monotonic()
            step_counter += 1

            all_tsp_empty = True
            for drone in drones:
                if hasattr(drone, 'tsp_plan') and len(drone.tsp_plan) > 0:
                    all_tsp_empty = False
                    break
            if all_tsp_empty:
                for drone in drones:
                    if getattr(drone, 'search_mode', None) == 'TSP':
                        drone.search_mode = 'GREEDY'
                        print(f"  [D{drone.id}] Switching from TSP to GREEDY mode because all TSP plans are empty")

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
                        
                elif drone.search_mode == 'GREEDY':
                    drone.final_action = drone.greedy_solver.calculate_greedy(
                        drone.belief_map, 
                        drone.pos, 
                        drone.partner_positions
                    )

            for drone in drones:
                drone.execute_move()

            for drone in drones:
                drone.get_real_observation(target_pos, traces)
            
            observation_packets = [drone.send_observation() for drone in drones]
            
            for drone in drones:
                for pkt in observation_packets:
                    if pkt['id'] != drone.id:
                        drone.receive_remote_observation(pkt['id'], pkt['pos'], pkt['observation'])
            
            for drone in drones:
                drone.update_belief_from_all_obs()

            render_frame(graphics_ctx, drones, target_pos, traces, step_counter)

            # Calcolo entropia mappa e aggiunta alla cronologia
            flat_belief = drones[0].belief_map.flatten()
            current_entropy = entropy(flat_belief, base=2)
            entropy_history.append(current_entropy)

            if drones[0].belief_map.max() >= 0.95:
                print(f"\n TARGET TROVATO in {step_counter} step! (probabilità > 95%)")
                pygame.quit()
                return step_counter, entropy_history
        
        graphics_ctx['clock'].tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit()
                return -1, []
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: 
                    pygame.quit()
                    return -1, []
                if event.key == pygame.K_SPACE: 
                    auto_mode = not auto_mode