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
    'RED': (255, 50, 50),
    'GREEN': (0, 200, 0), 
    'PURPLE': (200, 50, 255), 
    'ORANGE': (255, 140, 0),
    'LIGHT_BLUE': (0, 0, 255), 
    'LIGHT_RED': (200, 0, 0)
}

DARP_AREA_COLORS = [
    (144, 238, 144), 
    (255, 160, 160), 
    (216, 191, 216), 
    (173, 216, 230),
    (255, 255, 204), 
    (210, 180, 140), 
    (211, 211, 211), 
    (255, 218, 185)
]

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
# CENTRALIZED POMCP LOGIC 
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
        
        self.root_action_flips = 0
        self.last_root_action = None
        iterations = 0

        root = POMCPNode(belief_map=current_belief_map, parent=None)
        self.root = root 
        start_time = time.time()
        
        while True:
            if (time.time() - start_time) > self.max_time:
                break
            
            iterations += 1
            sampled_target_pos = self._sample_target_from_belief(root.belief_map)
            # Stato centrale: Target e tupla delle posizioni di tutti i droni
            state = (sampled_target_pos, tuple(drone_positions))
            self.simulate(state, root, 0, visited_cells=None, current_visited_cells=current_visited_cells)
        
        best_action = self._select_best_action(root)
        
        # Excel metrics calculation
        exploitation = root.q_value_actions.get(best_action, 0.0)
        n_a = root.action_counts.get(best_action, 0)
        N = root.total_node_visits
        
        exploration = 0.0
        if n_a > 0 and N > 0:
            exploration = self.c * math.sqrt(math.log(N) / n_a)
            
        expl_ratio = 0.0
        if (exploration + abs(exploitation)) != 0:
            expl_ratio = (exploration / (exploration + abs(exploitation))) * 100

        metrics = {
            'iterations': iterations,
            'depth': self.max_depth_reached,
            'nodes': self.total_nodes_created,
            'expl_ratio': expl_ratio,
            'flips': self.root_action_flips
        }

        return best_action, metrics

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
            if not node.action_counts: 
                return -100.0
            rollout_value = self.rollout(state)
            node.total_node_visits += 1
            return rollout_value

        action = self._ucb_search(node)
        
        # Root flips tracking
        if node == self.root:
            if self.last_root_action is not None and action != self.last_root_action:
                self.root_action_flips += 1
            self.last_root_action = action

        next_state, observation, reward, terminal = self.generative_model_G(state, action, node.belief_map, visited_cells, current_visited_cells)

        if (action, observation) in node.children:
            child_node = node.children[(action, observation)]
        else:
            _, next_drones_pos = next_state
            # Bayesian update 
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
        
        # Generate all possible joint actions
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
            
            # Check collisions 
            if len(set(next_positions)) != self.num_drones:
                continue
                
            # Check swaps
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
# CENTRALIZED BELIEF FUSION & AGENTS
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


# =============================================================================
# DRONE AGENT 
# =============================================================================

class DroneAgent:
    def __init__(self, drone_id, start_pos, params):
        self.id = drone_id
        self.pos = start_pos
        self.params = params
        self.observation = None
        self.positive_obs_count = 0
        self.tsp_plan = deque(self.params.get('tsp_plans', {}).get(self.id, []))

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


# =============================================================================
# GRAPHIC FUNCTIONS 
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

def draw_static_background(graphics_ctx, global_belief_map, drones, global_mode):
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

    for r in range(map_size):
        for c in range(map_size):
            x = c * cell_size
            y = r * cell_size
            prob = global_belief_map[r, c]

            if obstacle_map is not None and obstacle_map[r, c] == 1:
                color = COLORS['BLACK']
            else:
                if global_mode == 'TSP' and darp_matrix is not None and (darp_matrix[r, c] != -1 or (r, c) in drone_start_positions):
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

    # Fixed elements at the bottom
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

def render_frame(graphics_ctx, global_belief_map, drones, target_pos, traces, ui_stats, global_mode):
    draw_static_background(graphics_ctx, global_belief_map, drones, global_mode)
    graphics_ctx['screen'].fill(COLORS['WHITE'])
    graphics_ctx['screen'].blit(graphics_ctx['background_surface'], (0, 0))
    
    draw_elements(graphics_ctx, global_belief_map, drones, target_pos, traces, ui_stats)
    draw_tsp_paths(graphics_ctx, drones)
    
    pygame.display.flip()


# =============================================================================
# MAIN LOOP
# =============================================================================

def run_simulation(params):
    use_gui = params.get('use_gui', True)
    
    if use_gui:
        graphics_ctx = init_graphics(params)
    else:
        graphics_ctx = None
    
    num_drones = params['num_drones']
    target_pos = params['target_pos']
    traces = params['traces']
    obstacle_map = initialize_obstacle_map(params)
    dist_BFS = params.get('dist_BFS', precompute_BFS_distances(params['map_size'], obstacle_map))
    
    global_belief_map = initialize_belief_map(params)
    global_discovered_traces = set()
    global_explored_cells = set()
    
    global_mode = 'TSP' if params['map_type'] == 1 else 'POMCP'
    
    drones = [DroneAgent(i+1, params['drone_positions'][i], params) for i in range(num_drones)]
    centralized_solver = POMCPSolver(params, obstacle_map, dist_BFS)

    running = True
    auto_mode = True 
    step_counter = 0
    move_interval_sec = 0.0 
    last_step_time = 0.0
    
    # List for metrics 
    centralized_metrics = []

    while running:
        
        if not use_gui or (auto_mode and (time.monotonic() - last_step_time) >= move_interval_sec):
            
            if use_gui:
                last_step_time = time.monotonic()
                
            step_counter += 1
            
            if global_mode == 'TSP' and all(len(d.tsp_plan) == 0 for d in drones):
                global_mode = 'POMCP'
            
            joint_action = []
            current_metric = None 
            
            # Actions planning
            if global_mode == 'TSP':
                for drone in drones:
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
                        
                current_metric = None

            elif global_mode == 'POMCP':
                current_positions = [d.pos for d in drones]

                joint_action, current_metric = centralized_solver.search(global_belief_map.copy(), current_positions, global_explored_cells)

            centralized_metrics.append(current_metric)

            # Physical execution of actions and observation collection
            for i, drone in enumerate(drones):
                drone.execute_move(joint_action[i])
                drone.get_real_observation(target_pos, traces, global_discovered_traces)
                global_explored_cells.add(drone.pos)

            # Centralized belief fusion
            for drone in drones:
                obs = drone.observation
                
                if isinstance(obs, dict) and 'type' in obs:
                    if obs['pos'] not in global_discovered_traces:
                        global_belief_map = apply_trace_distribution(global_belief_map, obs, params, obstacle_map)
                        global_discovered_traces.add(obs['pos'])
                        
                        if global_mode == 'TSP':
                            global_mode = 'POMCP'
                            for d in drones: 
                                d.tsp_plan.clear()
                
                elif isinstance(obs, int):
                    global_belief_map = get_updated_belief_map_with_sensors(
                        global_belief_map, drone.pos, obs, params['alpha_sensor'], params['beta_sensor'], obstacle_map
                    )

            # Winning condition check
            if global_belief_map.max() >= 0.95:
                print(f"\nTarget found in {step_counter} steps!")
                print("------------------------------------------------")
                if use_gui:
                    pygame.quit()
                return step_counter, None, centralized_metrics
            
            # Safety check to prevent infinite loops
            if step_counter >= 180:
                if use_gui:
                    pygame.quit()
                return 180, None, centralized_metrics
            
        if use_gui:
            ui_stats = {
                'step': step_counter,
                'drones': {}
            }
            
            for i, drone in enumerate(drones):
                current_action = '-'
                if global_mode == 'POMCP' and 'joint_action' in locals() and joint_action:
                    current_action = joint_action[i]
                elif global_mode == 'TSP' and drone.tsp_plan:
                    current_action = drone.tsp_plan[0] if len(drone.tsp_plan) > 0 else 'Stay'

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

            render_frame(graphics_ctx, global_belief_map, drones, target_pos, traces, ui_stats, global_mode)
            graphics_ctx['clock'].tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    pygame.quit()
                    return -1, [], []
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: 
                        pygame.quit()
                        return -1, [], []
                    if event.key == pygame.K_SPACE: 
                        auto_mode = not auto_mode
                        if auto_mode:
                            last_step_time = time.monotonic() - move_interval_sec