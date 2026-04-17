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

#Global constants
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

# Functions to initialize map 
def initialize_obstacle_map(params):
    map_size = params['map_size']
    obstacle_map = np.zeros((map_size, map_size), dtype=int)
    for obs_pos in params.get('obstacles', []):
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
# 2. DEC-POMCP LOGIC 
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
    def __init__(self, max_time, depth_limit, discount_factor, exploration_const, sensor_alpha, sensor_beta, reward_alpha, explorative_reward, map_size, obstacle_map, drone_id, dist_BFS, r_target, explored_cells):
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
        self.dist_BFS = dist_BFS

    def search(self, current_belief_map, drone_position, partner_positions=None, partner_plans=None):
        self.drone_position = drone_position
        self.partner_positions = partner_positions
        self.partner_plans = partner_plans
        self.total_nodes_created = 1 
        self.max_depth_reached = 0

        # VARIABILI NUOVE PER TRACCIAMENTO
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
            state = (sampled_target_pos, self.drone_position)
            self.simulate(state, root, 0, visited_cells=None, current_visited_cells=self.explored_cells)
        
        best_action = self._select_best_action(root)
        future_plans = self._extract_future_plans(root)
        
        # --- CALCOLO METRICHE PER EXCEL ---
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
        
        return best_action, future_plans, metrics

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
            is_root = (node.parent is None) 
            self.expand(node, state, self.partner_positions if is_root else None)
            rollout_value = self.rollout(state)
            node.total_node_visits += 1 
            return rollout_value

        action = self._ucb_search(node)
        
        # TRACCIAMENTO FLIP AZIONE AL ROOT
        if node == self.root:
            if self.last_root_action is not None and action != self.last_root_action:
                self.root_action_flips += 1
            self.last_root_action = action

        next_state, observation, reward, terminal = self.generative_model_G(state, action, node.belief_map, visited_cells, current_visited_cells, depth)

        if (action, observation) in node.children:
            child_node = node.children[(action, observation)]
        else:
            _, next_drone_pos = next_state
            new_belief_map = self.get_updated_belief_map_with_sensors(node.belief_map, next_drone_pos, observation, self.sensor_alpha, self.sensor_beta)
            child_node = POMCPNode(belief_map=new_belief_map, parent=node)
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

    def expand(self, node, state, partner_positions=None):
        _, drone_pos = state 
        partner_list = []
        if partner_positions is not None:
            partner_list = list(partner_positions.values())

        for action in MOVES_DELTA.keys():
            delta = MOVES_DELTA[action]
            next_pos = (drone_pos[0] + delta[0], drone_pos[1] + delta[1])

            if not (0 <= next_pos[0] < self.map_size and 0 <= next_pos[1] < self.map_size):
                continue
            if self.obstacle_map[next_pos[0], next_pos[1]] == 1:
                continue
            if partner_list and next_pos in partner_list:
                continue
            
            node.action_counts[action] = 0         
            node.q_value_actions[action] = 0.0     

    def rollout(self, state):
        target_pos, drone_pos = state
        dist = self.dist_BFS.get((drone_pos, target_pos))
        if dist is None:
            return 0.0 
        score = (0.75 ** dist)
        return score

    def generative_model_G(self, state, action, belief_map, visited_cells, current_visited_cells, depth=0):
        target_pos, drone_pos = state 
        delta = MOVES_DELTA[action]
        next_drone = (drone_pos[0] + delta[0], drone_pos[1] + delta[1])
        next_state = (target_pos, next_drone)

        if (next_drone == target_pos):
            obs = 0 if np.random.rand() < self.sensor_beta else 1
        else:
            obs = 1 if np.random.rand() < self.sensor_alpha else 0
                      
        terminal = False        
        if (next_drone == target_pos) and obs == 1:
            r_target_reward = self.r_target          
            terminal = True     
        else:
            r_target_reward = 0.0          

        if next_drone not in visited_cells:
            r_token = belief_map[next_drone]        
        else:
            r_token = 0.0                            

        base_reward = r_target_reward + (self.reward_alpha * r_token)
        explorative_bonus = self.explorative_reward if next_drone not in current_visited_cells else 0.0
        
        visited_cells.add(next_drone)
        current_visited_cells.add(next_drone)

        total_reward = base_reward + explorative_bonus
        return next_state, obs, total_reward, terminal

    def get_updated_belief_map_with_sensors(self, current_belief, drone_pos, observation, alpha_sensor, beta_sensor):
        if observation == 1:
            Psi = 1.0 - beta_sensor 
            Phi = alpha_sensor       
        else:
            Psi = beta_sensor        
            Phi = 1.0 - alpha_sensor 

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

    def build_virtual_belief_map(self, current_belief_map, my_pos, partner_positions, partner_plans):
        virtual_belief_map = current_belief_map.copy()

        if not partner_positions or not partner_plans:
            return virtual_belief_map

        simulated_positions = {}
        for partner_id, partner_pos in partner_positions.items():
            dist_manhattan = abs(my_pos[0] - partner_pos[0]) + abs(my_pos[1] - partner_pos[1])
            if not (dist_manhattan <= 5 and partner_id > self.drone_id):
                simulated_positions[partner_id] = partner_pos

        if not simulated_positions:
            return virtual_belief_map

        # Apply negative observation update for current positions of partners
        for partner_id, current_pos in simulated_positions.items():
            virtual_belief_map = self.get_updated_belief_map_with_sensors(virtual_belief_map, current_pos, 0, self.sensor_alpha, self.sensor_beta)

        max_horizon = 0
        for partner_id, plan in partner_plans.items():
            if partner_id in simulated_positions and plan:
                max_horizon = max(max_horizon, len(plan))

        if max_horizon == 0:
            return virtual_belief_map

        for step_idx in range(max_horizon):
            progress = 0.0 if max_horizon == 1 else step_idx / (max_horizon - 1)
            effective_alpha = self.sensor_alpha + (0.1 - self.sensor_alpha) * progress
            effective_beta = self.sensor_beta + (0.1 - self.sensor_beta) * progress

            cells_to_update = set()

            for partner_id, current_pos in list(simulated_positions.items()):
                plan = partner_plans.get(partner_id, [])
                if step_idx >= len(plan):
                    continue

                action = plan[step_idx]
                delta = MOVES_DELTA.get(action, (0, 0))
                next_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])

                simulated_positions[partner_id] = next_pos
                cells_to_update.add(next_pos)

            for cell in cells_to_update:
                virtual_belief_map = self.get_updated_belief_map_with_sensors(virtual_belief_map, cell, 0, effective_alpha, effective_beta)

        return virtual_belief_map

    def _sample_target_from_belief(self, belief_map):
        flat_probs = belief_map.flatten()
        total = np.sum(flat_probs)
        if abs(total - 1.0) > 1e-6: 
            if total > 1e-9:
                flat_probs = flat_probs / total
            else:
                flat_probs = np.ones_like(flat_probs) / flat_probs.size
        
        indices = np.arange(belief_map.size)
        sampled_idx = np.random.choice(indices, p=flat_probs)
        x, y = np.unravel_index(sampled_idx, belief_map.shape)
        return (x, y)

    def _ucb_search(self, node):
        best_val = -float('inf') 
        ucb_best_action = None 

        log_total_visits = math.log(node.total_node_visits) if node.total_node_visits > 0 else 0 
        infinite_actions = []

        for action in node.action_counts.keys():
            n_ba = node.action_counts[action]
            q_ba = node.q_value_actions[action]

            if n_ba == 0:
                uct_val = float('inf')
                infinite_actions.append(action) 
            else:
                uct_val = q_ba + self.c * math.sqrt(log_total_visits / n_ba) 

            if uct_val > best_val:
                best_val = uct_val
                ucb_best_action = action

        if infinite_actions:
            return random.choice(infinite_actions) 

        return ucb_best_action

    def _select_best_action(self, node):
        if not node.q_value_actions:
            return 'Stay'
        best_action = max(node.q_value_actions.items(), key=lambda x: x[1])[0]
        return best_action

    def _extract_future_plans(self, root):
        future_plans = {}
        for root_action in root.action_counts.keys():
            plan = [root_action] 
            matching_children = [(action_obs, child) for action_obs, child in root.children.items() if action_obs[0] == root_action]
            if not matching_children:
                future_plans[root_action] = [root_action]
                continue
            
            most_visited_child = max(matching_children, key=lambda x: x[1].total_node_visits)[1]
            current_node = most_visited_child
            
            for _ in range(5):
                if not current_node.q_value_actions or current_node.total_node_visits == 0:
                    break
                visited_actions = {action: q_val for action, q_val in current_node.q_value_actions.items() if current_node.action_counts.get(action, 0) > 0}
                if not visited_actions:
                    break
                best_action_here = max(visited_actions.items(), key=lambda x: x[1])[0]
                plan.append(best_action_here)
                matching_children_next = [(action_obs, child) for action_obs, child in current_node.children.items() if action_obs[0] == best_action_here]
                if not matching_children_next:
                    break
                current_node = max(matching_children_next, key=lambda x: x[1].total_node_visits)[1]
            
            future_plans[root_action] = plan
        return future_plans

# =============================================================================
# MULTIPROCESSING WORKER FUNCTION FOR POMCP
# =============================================================================

def worker_pomcp_task(params, belief_map, my_pos, partner_positions, partner_plans, drone_id, obstacle_map, explored_cells):
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
    virtual_belief_map = solver.build_virtual_belief_map(belief_map, my_pos, partner_positions, partner_plans)
    best_action, future_plans, metrics = solver.search(virtual_belief_map, my_pos, partner_positions, partner_plans)
    return {
        'best_action': best_action,
        'depth': solver.max_depth_reached,
        'visits': solver.root.total_node_visits,
        'nodes_created': solver.total_nodes_created,
        'future_plans': future_plans,
        'metrics': metrics
    }

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
        self.search_mode = 'TSP' if params['map_type'] == 1 else 'POMCP'    
        self.belief_map = initialize_belief_map(params) 
        self.explored_cells = set()     
        self.obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else np.zeros((params['map_size'], params['map_size']), dtype=int) 
        
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

        self.planned_result = None      
        self.pos = start_pos            
        self.final_action = None        
        self.future_plans_buffer = {}   
        self.observation = None         
        self.positive_obs_count = 0     

        self.partner_positions = partner_positions if partner_positions is not None else {}     
        self.partner_final_actions = {}     
        self.partner_future_plans = {}      
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

    def send_intention(self):
        return {'id': self.id, 'pos': self.pos, 'best_action': self.planned_result.get('best_action', 'Stay')}

    def receive_intention(self, drone_id, position, best_action):
        self.partner_final_actions[drone_id] = best_action
        self.partner_positions[drone_id] = position

    def resolve_conflicts_local(self):
        all_drones_info = {self.id: (self.pos, self.final_action)}     
        for partner_id, partner_action in self.partner_final_actions.items():
            partner_pos = self.partner_positions[partner_id]
            all_drones_info[partner_id] = (partner_pos, partner_action)
        
        future_positions = {}
        for drone_id, (pos, action) in all_drones_info.items():
            delta = MOVES_DELTA[action]
            future_pos = (pos[0] + delta[0], pos[1] + delta[1])
            future_positions[drone_id] = future_pos
        
        pos_to_drones = {}
        for drone_id, future_pos in future_positions.items():
            if future_pos not in pos_to_drones:
                pos_to_drones[future_pos] = []
            pos_to_drones[future_pos].append(drone_id)
        
        final_actions = {}
        for future_pos, drone_ids in pos_to_drones.items():
            if len(drone_ids) > 1: 
                drone_ids_sorted = sorted(drone_ids) 
                winner_id = drone_ids_sorted[0]
                for drone_id in drone_ids:
                    if drone_id == winner_id:
                        final_actions[drone_id] = all_drones_info[drone_id][1] 
                    else:
                        final_actions[drone_id] = 'Stay' 
            else:
                drone_id = drone_ids[0]
                final_actions[drone_id] = all_drones_info[drone_id][1] 
        
        original_action = self.final_action
        self.final_action = final_actions[self.id]
        
        for partner_id in self.partner_final_actions.keys():
            self.partner_final_actions[partner_id] = final_actions[partner_id]
        
        if self.final_action != original_action:
            pass # debug silenziato per velocità

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

    def send_observation_and_future_plan(self):
        executed_plan = self.future_plans_buffer.get(self.final_action, [])     
        future_plan = executed_plan[1:] if len(executed_plan) > 1 else []       
        return {
            'id': self.id,
            'pos': self.pos,
            'observation': self.observation,
            'future_plan': future_plan
        }

    def receive_remote_observation(self, drone_id, position, observation, future_plan):
        self.partner_positions[drone_id] = position
        self.partner_observations[drone_id] = observation
        self.partner_future_plans[drone_id] = future_plan

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
                        self.search_mode = 'POMCP'
                        print(f"  [D{self.id}] Switching from TSP to POMCP mode due to trace detection")
            elif isinstance(obs, int) and obs in [0, 1]:    
                self.belief_map = self.solver_tool.get_updated_belief_map_with_sensors(self.belief_map, pos, obs, self.solver_tool.sensor_alpha, self.solver_tool.sensor_beta)
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

    is_pomcp_mode = False
    if drones is not None:
        is_pomcp_mode = any(getattr(drone, 'search_mode', None) == 'POMCP' for drone in drones)

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

def render_frame(graphics_ctx, drones, target_pos, traces, ui_stats):
    draw_static_background(graphics_ctx, drones[0].belief_map, drones)
    graphics_ctx['screen'].fill(COLORS['WHITE'])
    graphics_ctx['screen'].blit(graphics_ctx['background_surface'], (0, 0))
    draw_elements(graphics_ctx, drones, target_pos, traces, ui_stats)
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

    pool = multiprocessing.Pool(processes=num_drones)
    
    scenario_idx = params.get('scenario_idx', 1)
    algo_name = params.get('algo_name', 'POMCP')
    
    render_frame(graphics_ctx, drones, target_pos, traces, ui_stats)
    screenshot_name = f"Scenario_{scenario_idx}_{algo_name}_Frame0.png"
    pygame.image.save(graphics_ctx['screen'], screenshot_name)
    print(f"\n✓ Screenshot iniziale salvato come '{screenshot_name}'")

    running = True
    auto_mode = True
    step_counter = 0
    move_interval_sec = 0.0
    last_step_time = 0.0

    all_drones_metrics = {drone.id: [] for drone in drones}

    try:
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
                            drone.search_mode = 'POMCP'
                            print(f"  [D{drone.id}] Switching from TSP to POMCP mode because all TSP plans are empty")

                tasks = []
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
                        
                        drone.planned_result = {
                            'best_action': drone.final_action,
                            'depth': 0,
                            'visits': 0,
                            'nodes_created': 0,
                            'future_plans': {},
                            'metrics': None
                        }
                        drone.future_plans_buffer = {}

                    elif drone.search_mode == 'POMCP':
                        task = (drone.params, drone.belief_map.copy(), drone.pos, drone.partner_positions.copy(), drone.partner_future_plans.copy(), drone.id, drone.obstacle_map, drone.explored_cells.copy())
                        tasks.append(task)
                
                if tasks:
                    results = pool.starmap(worker_pomcp_task, tasks)
                    for i, drone in enumerate(drones):
                        if drone.search_mode == 'POMCP':
                            idx = [t[5] for t in tasks].index(drone.id)
                            drone.planned_result = results[idx]
                            drone.final_action = results[idx]['best_action']
                            drone.future_plans_buffer = results[idx]['future_plans']

                    intention_packets = [drone.send_intention() for drone in drones]
                    for drone in drones:
                        for pkt in intention_packets:
                            if pkt['id'] != drone.id:
                                drone.receive_intention(pkt['id'], pkt['pos'], pkt['best_action'])
                    
                    for drone in drones:
                        drone.resolve_conflicts_local()

                for drone in drones:
                    drone.execute_move()

                for drone in drones:
                    drone.get_real_observation(target_pos, traces)
                
                observation_packets = [drone.send_observation_and_future_plan() for drone in drones]
                for drone in drones:
                    for pkt in observation_packets:
                        if pkt['id'] != drone.id:
                            drone.receive_remote_observation(pkt['id'], pkt['pos'], pkt['observation'], pkt['future_plan'])
                
                for drone in drones:
                    drone.update_belief_from_all_obs()

                ui_stats['step'] = step_counter
                for drone in drones:
                    result = drone.planned_result
                    had_conflict = (drone.final_action != result.get('best_action', 'Stay'))
                    ui_stats['drones'][drone.id].update({
                        'obs': drone.observation,
                        'depth': result.get('depth', 0),
                        'visits': result.get('visits', 0),
                        'nodes': result.get('nodes_created', 0),
                        'best': result.get('best_action', '-'),
                        'final': drone.final_action,
                        'conflict': had_conflict
                    })

                    metric = result.get('metrics', None)
                    all_drones_metrics[drone.id].append(metric)

                render_frame(graphics_ctx, drones, target_pos, traces, ui_stats)

                if drones[0].belief_map.max() >= 0.95:
                    print(f"\n TARGET TROVATO in {step_counter} step! (probabilità > 95%)")
                    pygame.quit()
                    return step_counter, None, all_drones_metrics

            graphics_ctx['clock'].tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    pygame.quit()
                    return -1, [], {}
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: 
                        pygame.quit()
                        return -1, [], {}
                    if event.key == pygame.K_SPACE: 
                        auto_mode = not auto_mode
            
    finally:
        pool.close()    
        pool.join()
