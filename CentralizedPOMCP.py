import math
import random
import time
import numpy as np
from scipy.stats import multivariate_normal
import pygame
import imageio
from datetime import datetime


# Configurazione valori default
DEFAULT_CONFIG = {
    'map_size': 15,
    'real_alpha': 0.01,
    'real_beta': 0.01,
    'max_iterations': 1000000,
    'max_time': 3.5,
    'depth_limit': 100,
    'discount_factor': 0.9,
    'exploration_const': math.sqrt(2),
    'reward_alpha': 3,
}

#Funzione per raccolta parametri utente
def get_user_parameters():
    
    print("=== CONFIGURAZIONE MISSIONE DI RICERCA ===")
    
    cfg = DEFAULT_CONFIG
    map_size = cfg['map_size']
    real_alpha = cfg['real_alpha']
    real_beta = cfg['real_beta']
    print(f"Parametri: Map Size={map_size}x{map_size}, Alpha={real_alpha}, Beta={real_beta}")
    
    # Funzione di sicurezza per inserire coordinate valide
    def _get_coord(name):
        while True:
            try:
                val = input(f"Inserisci coordinate {name} (formato 'riga,colonna'): ")
                r, c = map(int, val.split(','))
                if 0 <= r < map_size and 0 <= c < map_size:
                    return (r, c)
                else:
                    print(f"Errore: coordinate devono essere tra 0 e {map_size-1}")
            except ValueError:
                print("Errore: formato non valido. Usa 'riga,colonna'")

    #Input Posizioni Droni e Target
    d1_pos = _get_coord("Drone 1")
    d2_pos = _get_coord("Drone 2")
    target_pos = _get_coord("Target Reale")
    
    #Scelta della distribuzione iniziale sulla mappa
    map_config = {}
    print("\n--- Configurazione Belief Map Iniziale ---")
    print("1. Uniforme")
    print("2. Singola Gaussiana (1 picco)")
    print("3. Multi-Gaussiana (N picchi)")
    
    while True:
        try:
            choice = int(input("Scegli il tipo di mappa (1-3): "))
            if choice in [1, 2, 3]:
                break
            print("Scelta non valida.")
        except ValueError:
            print("Inserisci un numero.")

    map_config['type'] = choice
    map_config['peaks'] = []

    #Acquisizione parametri per gaussiane
    if choice == 2 or choice == 3:
        num_peaks = 1 if choice == 2 else 0
        
        if choice == 3:
            while True:
                try:
                    num_peaks = int(input("Quanti picchi vuoi inserire? "))
                    if num_peaks > 0: break
                except ValueError: pass
        
        for i in range(num_peaks):
            print(f"\nConfigurazione Picco #{i+1}:")

            # Centro (Media)
            mu = _get_coord(f"centro della gaussiana #{i+1}")
            
            # Deviazioni Standard 
            while True:
                try:
                    sigmas = input("Inserisci deviazione standard Sigma_X, Sigma_Y : ")
                    sx, sy = map(float, sigmas.split(','))
                    if sx > 0 and sy > 0:
                        break
                    print("Le deviazioni standard devono essere positive.")
                except ValueError:
                    print("Formato errato.")
            
            map_config['peaks'].append({
                'mean': mu,
                'cov': [sx, sy] # Usiamo lista per costruire matrice diagonale poi
            })

    # Restituzione dizionario parametri
    return {
        'map_size': map_size,
        'real_alpha': real_alpha,
        'real_beta': real_beta,
        'max_iterations': cfg['max_iterations'],
        'max_time': cfg['max_time'],
        'depth_limit': cfg['depth_limit'],
        'discount_factor': cfg['discount_factor'],
        'exploration_const': cfg['exploration_const'],
        'reward_alpha': cfg['reward_alpha'],
        'd1_pos': d1_pos,
        'd2_pos': d2_pos,
        'target_pos': target_pos,
        'map_config': map_config
    }

# Funzione per generazione della mappa di probabilità
def initialize_belief_map(params):

    map_size = params['map_size']
    config = params['map_config']
    
    # Inizializziamo mappa vuota
    belief_map = np.zeros((map_size, map_size))
    
    # Caso 1: Uniforme
    if config['type'] == 1:
        belief_map.fill(1.0) # Riempie tutto di 1, poi normalizzeremo
        
    # Caso 2 e 3: Gaussiane
    else:
        # Creazione griglia per distribuzione pdf
        x, y = np.mgrid[0:map_size, 0:map_size]
        coord = np.dstack((x, y))
        
        for peak in config['peaks']:
            mean = peak['mean']     # (riga, colonna)
            sigmas = peak['cov']    # [sx, sy]
            
            # Matrice di covarianza (diagonale per semplicità)
            cov_matrix = [[sigmas[0]**2, 0], [0, sigmas[1]**2]]
            
            # Creiamo l'oggetto multivariata
            rv = multivariate_normal(mean, cov_matrix)
                                    
            # Aggiunta PDF alla mappa + discretizzazione della probabilità
            belief_map += rv.pdf(coord)

    # Normalizzazione della mappa + check somma di tutte le celle deve fare 1.0 
    total_prob = np.sum(belief_map)
    
    if total_prob == 0:
        # Fallback di sicurezza per valori di varianze enormi
        belief_map.fill(1.0 / (map_size * map_size))
    else:
        belief_map /= total_prob
        
    return belief_map

#Nodo dell'albero POMCP: contiene mappa di probabilità e azioni (in particolare passiamo nodo padre)
class POMCPNode:
   
    def __init__(self, belief_map, parent=None):
        self.belief_map = belief_map  
        self.parent = parent
        
        # N(b): numero di visite al nodo
        self.visits = 0  
        
        # Collegamento tra nodo attuale e nodi figli
        # children key: (action, observation) -> value: POMCPNode
        self.children = {} 
        
        # Qualità media da questo nodo per ogni azione Q(b,a)
        # value_estimates key: action -> value: Q(b, a) (Valore medio)
        self.value_estimates = {} 
        
        # Numero di volte che ogni azione "a" è stata eseguita da questo nodo N(b,a)
        # action_counts key: action -> value: N(b, a)
        self.action_counts = {}

    # check se il nodo è foglia
    def is_leaf(self):
                        
        return self.visits == 0

#Classe principale del solver POMCP
class POMCPSolver:
    def __init__(self, max_iterations=None, max_time=None, depth_limit=None, discount_factor=None,
                 exploration_const=None, sensor_alpha=None, sensor_beta=None,
                 reward_alpha=None, map_size=None):
        cfg = DEFAULT_CONFIG
        self.max_iterations = max_iterations if max_iterations is not None else cfg['max_iterations']
        self.max_time = max_time if max_time is not None else cfg['max_time']
        self.depth_limit = depth_limit if depth_limit is not None else cfg['depth_limit']
        self.gamma = discount_factor if discount_factor is not None else cfg['discount_factor']
        self.c = exploration_const if exploration_const is not None else cfg['exploration_const']
        self.sensor_alpha = sensor_alpha if sensor_alpha is not None else cfg['real_alpha']
        self.sensor_beta = sensor_beta if sensor_beta is not None else cfg['real_beta']
        self.reward_alpha = reward_alpha if reward_alpha is not None else cfg['reward_alpha']
        self.map_size = map_size if map_size is not None else cfg['map_size']

        self.total_nodes_created = 0  # Contatore nodi creati durante search
        self.max_depth_reached = 0    # Profondità massima raggiunta
        self.last_top_actions = []    # Top azioni dell'ultima ricerca

    # Funzione di ricerca POMCP: costruiamo albero + restituzione azione migliore finale
    def search(self, current_belief_map, drone_positions):
        
        # Creazione del nodo radice con la belief map attuale
        root = POMCPNode(belief_map=current_belief_map)
        self.root = root  
        self.total_nodes_created = 1  
        self.max_depth_reached = 0
        self.last_top_actions = []
        
        start_time = time.time()
        
        # Ciclo principale di simulazione Monte Carlo
        for i in range(self.max_iterations):
            if (time.time() - start_time) > self.max_time:
                break
            
            # Campionamento stato iniziale: estrazione posizione target ad ogni iterazione (stato: pos droni e pos target)
            sampled_target_pos = self._sample_target_from_belief(root.belief_map)
            state = (sampled_target_pos, drone_positions[0], drone_positions[1])
            
            # Avvio della simulazione ricorsiva 
            self.simulate(state, root, 0)
        
        # Stampa statistiche albero
        print(f"\n[POMCP] Nodo root visitato {root.visits} volte")
        print(f"[POMCP] Azioni esplorate: {len(root.action_counts)}")
        print(f"[POMCP] Nodi totali creati: {self.total_nodes_created}")

        action_stats = []
        for action, count in root.action_counts.items():
            q_val = root.value_estimates.get(action, 0.0)
            action_stats.append({
                'action': action,
                'q': q_val,
                'n': count
            })

        action_stats.sort(key=lambda entry: entry['q'], reverse=True)
        self.last_top_actions = action_stats[:3]
        
        # Selezione dell'azione migliore: azione con Q massimo
        best_action = self._select_best_action(root)
        return best_action

    # Singola simulazione POMCP: fatta in maniera ricorsiva per scendere in profondità
    def simulate(self, state, node, depth,visited_cells=None): 
        
        # Inizializzazione del set alla radice
        if visited_cells is None:
            visited_cells = set()

        # Aggiorna profondità massima raggiunta finora
        if depth > self.max_depth_reached:
            self.max_depth_reached = depth

        # Controllo terminazione (Depth o Stato Terminale: se target trovato in simulazione)
        if depth >= self.depth_limit:
            return 0.0

        # Espansione e Rollout 
        if node.is_leaf():
            # Se il nodo non ha figli, generiamo le azioni possibili
            self.expand(node, state)
            # Se dopo l'espansione non ci sono azioni valide (es. droni bloccati), ritorniamo penalty
            if not node.action_counts:
                return -100.0 # Penalty per stallo/vicolo cieco
            rollout_value = self.rollout(state)
            node.visits += 1  # Conta la visita al nodo foglia
            # Ritorniamo valore del rollout
            return rollout_value

        # Selezione Azione tramite UCT
        action = self._ucb_search(node)

        # Generative Model (G): simula transizione black box (s, a) -> (s', o, r)
        next_state, observation, reward, terminal = self.generative_model_G(state, action, node.belief_map, visited_cells)

        # Discesa nell'albero: verifica se esiste nodo figlio 
        if (action, observation) in node.children:
            child_node = node.children[(action, observation)]
        else:
            
            # Estrazione nuove posizioni dai next_state per aggiornare la mappa
            _, next_d1_pos, next_d2_pos = next_state
            # Calcoliamo la nuova belief map
            new_belief_map = self.get_updated_belief_map(node.belief_map, next_d1_pos, next_d2_pos, observation)
            
            child_node = POMCPNode(belief_map=new_belief_map, parent=node)
            node.children[(action, observation)] = child_node
            self.total_nodes_created += 1  

        # Ricorsione o Stop se Terminale 
        if terminal:
            future_reward = 0.0
        else:
            future_reward = self.simulate(next_state, child_node, depth + 1, visited_cells)
        q_value = reward + self.gamma * future_reward

        # Backpropagation: aggiorniamo N(b), N(b,a) e Q(b,a)
        node.visits += 1
        
        if action not in node.action_counts:
            node.action_counts[action] = 0
            node.value_estimates[action] = 0.0
            
        node.action_counts[action] += 1
        
        # Aggiornamento incrementale della media Q(b,a): Q_new = Q_old + (q_value - Q_old) / N(b,a)
        old_q = node.value_estimates[action]
        node.value_estimates[action] = old_q + (q_value - old_q) / node.action_counts[action]

        return q_value

    # Espansione nodo con combinazione azioni valide
    def expand(self, node, state):
        
        # Estrazioni posizioni attuali dei droni dallo stato
        _, d1_pos, d2_pos = state
        
        # Definizione mosse
        moves_delta = {
            'N': (-1, 0),
            'S': (1, 0),
            'W': (0, -1),
            'E': (0, 1),
            'Stay': (0, 0)
        }
        
        actions = list(moves_delta.keys())
        map_size = self.map_size  # Dimensione della griglia

        # Iteriamo su tutte le possibili combinazioni 
        for action_d1 in actions:
            for action_d2 in actions:
                
                # Calcolo posizione futura Drone 1
                delta1 = moves_delta[action_d1]
                next_d1 = (d1_pos[0] + delta1[0], d1_pos[1] + delta1[1])
                
                # Calcolo posizione futura Drone 2
                delta2 = moves_delta[action_d2]
                next_d2 = (d2_pos[0] + delta2[0], d2_pos[1] + delta2[1])
                
                # 1. Controllo Confini 
                if not (0 <= next_d1[0] < map_size and 0 <= next_d1[1] < map_size):
                    continue # Drone 1 esce dalla mappa
                
                if not (0 <= next_d2[0] < map_size and 0 <= next_d2[1] < map_size):
                    continue # Drone 2 esce dalla mappa

                # 2. Controllo Collisioni: droni scelgono stessa cella
                if next_d1 == next_d2:
                    continue 
                
                # 3. Controllo swapping posizione
                if next_d1 == d2_pos and next_d2 == d1_pos:
                    continue

                # Registrazione Azione Valida                 
                joint_action = (action_d1, action_d2)
                
                # Sicurezza: aggiungiamo azione se non esiste già
                if joint_action not in node.action_counts:
                    node.action_counts[joint_action] = 0
                    node.value_estimates[joint_action] = 0.0
                    # Nota: children non viene popolato qui, ma dopo aver osservato l'esito (step di simulate)

    # Rollout leggero basato su euristica di distanza di Manhattan
    def rollout(self, state):
        
        target_pos, d1_pos, d2_pos = state
        
        # 1. Calcolo distanza di Manhattan per il Drone 1
        dist_d1 = abs(target_pos[0] - d1_pos[0]) + abs(target_pos[1] - d1_pos[1])
        
        # 2. Calcolo distanza di Manhattan per il Drone 2
        dist_d2 = abs(target_pos[0] - d2_pos[0]) + abs(target_pos[1] - d2_pos[1])
        
        # Consideriamo solo il drone più vicino (minima distanza)
        min_dist = min(dist_d1, dist_d2)
        
        # 3. Calcolo del reward come funzione decrescente della distanza
        score = 1 * (self.gamma ** min_dist)
        return score

    #Black box simulator: transizione di stato(movimento droni), osservazione, reward
    def generative_model_G(self, state, action, belief_map, visited_cells):
        
        target_pos, d1_pos, d2_pos = state
        action_d1, action_d2 = action
        
        # 1. Transizione di Stato (Deterministica) 
        moves_delta = {
            'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1), 'Stay': (0, 0)
        }
        
        # Calcolo nuove posizioni (assumiamo che 'expand' abbia già validato i confini)
        d1_delta = moves_delta[action_d1]
        next_d1 = (d1_pos[0] + d1_delta[0], d1_pos[1] + d1_delta[1])
        
        d2_delta = moves_delta[action_d2]
        next_d2 = (d2_pos[0] + d2_delta[0], d2_pos[1] + d2_delta[1])
        
        # Il target non cambia posizione
        next_state = (target_pos, next_d1, next_d2)

        # 2. Generazione dell'Osservazione (Stocastica)
        #         
        # Osservazione Drone 1
        is_target_present_d1 = (next_d1 == target_pos)
        if is_target_present_d1:
            # Se il target c'è: rischio False Negative (beta)
            if np.random.rand() < self.sensor_beta:
                obs_d1 = 0
            else:
                obs_d1 = 1
        else:
            # Se il target NON c'è: rischio False Positive (alpha)
            if np.random.rand() < self.sensor_alpha:
                obs_d1 = 1
            else:
                obs_d1 = 0

        # Osservazione Drone 2
        is_target_present_d2 = (next_d2 == target_pos)
        if is_target_present_d2:
            if np.random.rand() < self.sensor_beta:
                obs_d2 = 0
            else:
                obs_d2 = 1
        else:
            if np.random.rand() < self.sensor_alpha:
                obs_d2 = 1
            else:
                obs_d2 = 0
        
        observation = (obs_d1, obs_d2)

        # 3. Calcolo della Reward 
        # Formula: R = R_target + reward_alpha * R_token                
        found_d1 = (is_target_present_d1 and obs_d1 == 1)
        found_d2 = (is_target_present_d2 and obs_d2 == 1)
        
        # A. R_target component
        terminal = False
        if found_d1 or found_d2:
            r_target = 1.0 # Target trovato
            terminal = True
        else:
            r_target = 0.0

        # B. R_token component
        r_token = 0.0
        for pos in [next_d1, next_d2]:
            # Controlliamo se la cella è già nel set delle visitate
            if pos not in visited_cells:
                
                if 0 <= pos[0] < self.map_size and 0 <= pos[1] < self.map_size:
                    r_token += belief_map[pos]
                
                # Aggiungiamo al set per non essere ricontata più avanti
                visited_cells.add(pos)
            else:
                # Ritorniamo 0 se cella già visitata
                r_token += 0.0

        total_reward = r_target + (self.reward_alpha * r_token)

        return next_state, observation, total_reward, terminal

    # Aggiornamento bayesiano della belief map con due sensori
    def get_updated_belief_map(self, current_belief, d1_pos, d2_pos, observation):
        
        obs_d1, obs_d2 = observation

        # 1. Update con osservazione Drone 1
        belief_mid = self._single_sensor_update(current_belief, d1_pos, obs_d1)

        # 2. Update con osservazione Drone 2 
        belief_final = self._single_sensor_update(belief_mid, d2_pos, obs_d2)

        return belief_final

    # Aggiornamento bayesiano per singolo sensore
    def _single_sensor_update(self, belief_map, inspected_cell, observation_Y):
        
        # 1. Definizione di Psi e Phi        
        if observation_Y == 1:
            # Positive Detection
            Psi = 1.0 - self.sensor_beta  # True Positive
            Phi = self.sensor_alpha       # False Positive
        else:
            # Negative Detection
            Psi = self.sensor_beta        # False Negative
            Phi = 1.0 - self.sensor_alpha # True Negative

        # 2. Calcolo termini intermedi
        Omega = Psi - Phi
        p_st = belief_map[inspected_cell]

        # 3. Calcolo del fattore di normalizzazione Z 
        Z = Phi + (Omega * p_st)

        # Protezione numerica per evitare divisione per zero
        if Z < 1e-9:
            return belief_map

        # 4. Calcolo del nuovo belief map 
        new_belief_map = (belief_map * Phi) / Z

        # Correzione della cella ispezionata 
        new_belief_map[inspected_cell] = (Psi * p_st) / Z

        return new_belief_map

    
    # Estrazione posizione target per POMCP
    def _sample_target_from_belief(self, belief_map):
        
        flat_probs = belief_map.flatten()
        indices = np.arange(belief_map.size)
        sampled_idx = np.random.choice(indices, p=flat_probs)
        x, y = np.unravel_index(sampled_idx, belief_map.shape)
        return (x, y)

    # Funzione UCB1 per selezione azione
    def _ucb_search(self, node):

        best_val = -float('inf')
        best_action = None 

        log_total_visits = math.log(node.visits) if node.visits > 0 else 0 
        infinite_actions = []  

        for action in node.action_counts.keys():
            n_ba = node.action_counts[action]
            q_ba = node.value_estimates[action]

            if n_ba == 0:
                uct_val = float('inf')
                infinite_actions.append(action)
            else:
                uct_val = q_ba + self.c * math.sqrt(log_total_visits / n_ba)

            if uct_val > best_val:
                best_val = uct_val
                best_action = action

        if infinite_actions:
            return random.choice(infinite_actions)

        if best_action is None and node.action_counts:
            best_action = random.choice(list(node.action_counts.keys()))

        return best_action
    
    
    # Selezione dell'azione migliore da eseguire nella realltà
    def _select_best_action(self, node):
    
        best_action = None
        best_val = -float('inf')
        
        for action, q_val in node.value_estimates.items():
            if q_val > best_val:
                best_val = q_val
                best_action = action
        return best_action


# === PARTE GRAFICA CON PYGAME ===

# Disegno griglia, heatmap e percentuali su sfondo
def draw_static_background(surface, p_map, font_cell, params):

    GRID_WIDTH = surface.get_width()
    CELL_SIZE = GRID_WIDTH // params['map_size']
    BLACK = (0, 0, 0)

    surface.fill((255, 255, 255))
    max_prob = p_map.max()

    for r in range(params['map_size']):
        for c in range(params['map_size']):
            prob = p_map[r, c]

            # Heatmap: blu più scuro = probabilità più alta
            color_val = 0
            if max_prob > 1e-9:
                color_val = int(255 * (prob / max_prob))
            color = (max(0, 255 - color_val), max(0, 255 - color_val), 255)

            # In Pygame: x = colonna (c), y = riga (r)
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)

            # Testo probabilità
            text = font_cell.render(f"{prob * 100:.3f}%", True, BLACK)
            surface.blit(text, (c * CELL_SIZE + 5, r * CELL_SIZE + 5))

# Funzioni per disegnare elementi dinamici: droni, target e barra laterale
def draw_elements(screen, belief_map, drones, params, font_sidebar, GRID_WIDTH, CELL_SIZE, steps_taken, pomcp_stats, SIDEBAR_WIDTH):
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 200, 0)
    GRAY = (200, 200, 200)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    PURPLE = (100, 0, 100)

    # Target (X) - posizione logica (riga, colonna) -> (x, y) Pygame
    tx, ty = params['target_pos']
    target_rect = pygame.Rect(ty * CELL_SIZE, tx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.line(screen, BLACK, target_rect.topleft, target_rect.bottomright, 3)
    pygame.draw.line(screen, BLACK, target_rect.topright, target_rect.bottomleft, 3)

    # Droni (cerchi) - posizioni logiche (riga, colonna)
    d1_pos = drones[0]
    d1_center = (d1_pos[1] * CELL_SIZE + CELL_SIZE // 2, d1_pos[0] * CELL_SIZE + CELL_SIZE // 2)
    pygame.draw.circle(screen, RED, d1_center, CELL_SIZE // 3, 4)

    d2_pos = drones[1]
    d2_center = (d2_pos[1] * CELL_SIZE + CELL_SIZE // 2, d2_pos[0] * CELL_SIZE + CELL_SIZE // 2)
    pygame.draw.circle(screen, GREEN, d2_center, CELL_SIZE // 3 - 4, 4)

    # Sidebar - estesa per tutta l'altezza della finestra
    screen_height = screen.get_height()
    sidebar_rect = pygame.Rect(GRID_WIDTH, 0, SIDEBAR_WIDTH, screen_height)
    pygame.draw.rect(screen, GRAY, sidebar_rect)

    # Statistiche
    y_offset = 20
    spacing = 22

    text_step = font_sidebar.render(f"Step: {steps_taken}", True, BLACK)
    screen.blit(text_step, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing + 8

    # Sezione POMCP Centralizzato
    text_pomcp_header = font_sidebar.render("=== POMCP Centralizzato ===", True, BLUE)
    screen.blit(text_pomcp_header, (GRID_WIDTH + 10, y_offset))
    y_offset += spacing

    text_depth = font_sidebar.render(f"Tree Depth: {pomcp_stats.get('depth', 0)}", True, BLACK)
    screen.blit(text_depth, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing

    text_sims = font_sidebar.render(f"Simulations: {pomcp_stats['simulations']}", True, BLACK)
    screen.blit(text_sims, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing

    text_nodes = font_sidebar.render(f"Nodes Created: {pomcp_stats['nodes']}", True, BLACK)
    screen.blit(text_nodes, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing

    text_time = font_sidebar.render(f"POMCP Time: {pomcp_stats['time']:.3f}s", True, BLACK)
    screen.blit(text_time, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing + 8

    # Azione migliore combinata
    text_best_header = font_sidebar.render("Best Joint Action:", True, BLACK)
    screen.blit(text_best_header, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing

    best_action = pomcp_stats.get('best_action', ('Stay', 'Stay'))
    best_q = pomcp_stats.get('best_q', 0.0)
    
    text_d1_action = font_sidebar.render(f"  Drone 1: {best_action[0]}", True, RED)
    screen.blit(text_d1_action, (GRID_WIDTH + 30, y_offset))
    y_offset += spacing

    text_d2_action = font_sidebar.render(f"  Drone 2: {best_action[1]}", True, GREEN)
    screen.blit(text_d2_action, (GRID_WIDTH + 30, y_offset))
    y_offset += spacing

    text_q_value = font_sidebar.render(f"  Q-value: {best_q:.4f}", True, BLACK)
    screen.blit(text_q_value, (GRID_WIDTH + 30, y_offset))
    y_offset += spacing + 8

    # Posizioni droni
    text_pos_header = font_sidebar.render("Drone Positions:", True, BLACK)
    screen.blit(text_pos_header, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing

    text_d1_pos = font_sidebar.render(f"  D1: {d1_pos}", True, RED)
    screen.blit(text_d1_pos, (GRID_WIDTH + 30, y_offset))
    y_offset += spacing

    text_d2_pos = font_sidebar.render(f"  D2: {d2_pos}", True, GREEN)
    screen.blit(text_d2_pos, (GRID_WIDTH + 30, y_offset))
    y_offset += spacing + 10

    # Max probabilità e cella
    max_prob = belief_map.max()
    max_pos = np.unravel_index(np.argmax(belief_map), belief_map.shape)
    text_max = font_sidebar.render(f"Max Prob: {max_prob:.4f}", True, BLACK)
    screen.blit(text_max, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing
    text_max_cell = font_sidebar.render(f"Max Cell: {max_pos}", True, BLACK)
    screen.blit(text_max_cell, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing + 10

    # Barra probabilità massima con threshold
    bar_width = SIDEBAR_WIDTH - 40
    pygame.draw.rect(screen, WHITE, (GRID_WIDTH + 20, y_offset, bar_width, 20))
    pygame.draw.rect(screen, BLUE, (GRID_WIDTH + 20, y_offset, bar_width * min(max_prob, 1.0), 20))
    thr_pos = (GRID_WIDTH + 20) + bar_width * 0.95
    pygame.draw.line(screen, GREEN, (thr_pos, y_offset - 3), (thr_pos, y_offset + 23), 2)
    text_thr = font_sidebar.render("Threshold 0.95", True, GREEN)
    screen.blit(text_thr, (GRID_WIDTH + 20, y_offset + 25))
    y_offset += 50

    # Top 3 azioni (opzionale - per vedere alternative)
    if pomcp_stats.get('top_actions', []):
        text_top_header = font_sidebar.render("Top Actions:", True, BLACK)
        screen.blit(text_top_header, (GRID_WIDTH + 20, y_offset))
        y_offset += spacing

        for idx, entry in enumerate(pomcp_stats.get('top_actions', [])[:3], start=1):
            action_str = f"{idx}. {entry['action']}"
            text_action = font_sidebar.render(action_str, True, BLACK)
            screen.blit(text_action, (GRID_WIDTH + 30, y_offset))
            y_offset += spacing

            detail_label = f"   Q={entry['q']:.4f}, N={entry['n']}"
            text_detail = font_sidebar.render(detail_label, True, BLACK)
            screen.blit(text_detail, (GRID_WIDTH + 40, y_offset))
            y_offset += spacing + 4

        y_offset += 10

    # Controlli
    text_auto = font_sidebar.render("SPAZIO: Auto Mode", True, BLACK)
    screen.blit(text_auto, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing

    text_gif = font_sidebar.render("G: REC/STOP GIF", True, PURPLE)
    screen.blit(text_gif, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing

    text_restart = font_sidebar.render("R: Riavvia", True, BLACK)
    screen.blit(text_restart, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing

    text_quit = font_sidebar.render("ESC: Esci", True, BLACK)
    screen.blit(text_quit, (GRID_WIDTH + 20, y_offset))

# === FUNZIONI PER SIMULARE REALTA' 

# Simula sensore reale
def get_real_observation(drone_positions, target_pos, alpha, beta):
    obs = []
    for pos in drone_positions:
        is_target = (pos == target_pos)
        if is_target:
            val = 0 if np.random.rand() < beta else 1
        else:
            val = 1 if np.random.rand() < alpha else 0
        obs.append(val)
    return tuple(obs)

# Movimento
def execute_move(current_positions, joint_action):
    moves_delta = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1), 'Stay': (0, 0)}
    new_positions = []
    for i, action in enumerate(joint_action):
        curr_r, curr_c = current_positions[i]
        delta_r, delta_c = moves_delta[action]
        new_r, new_c = curr_r + delta_r, curr_c + delta_c
        new_positions.append((new_r, new_c))
    return new_positions


# === MAIN LOOP CON PYGAME ===

def run_simulation(params):
    pygame.init()

    # Setup schermo
    map_size = params['map_size']
    cell_size = 60
    sidebar_w = 480
    GRID_WIDTH = map_size * cell_size
    screen_w = GRID_WIDTH + sidebar_w
    # Altezza minima per contenere tutti gli elementi della sidebar
    min_height = 750
    screen_h = max(map_size * cell_size, min_height)

    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Simulatore POMCP Drone Search")

    font_cell = pygame.font.SysFont(None, 18)
    font_sidebar = pygame.font.SysFont(None, 20)

    # Setup stato
    belief_map = initialize_belief_map(params)
    drone_positions = [params['d1_pos'], params['d2_pos']]
    target_pos = params['target_pos']

    # Solver
    solver = POMCPSolver(
        max_iterations=params['max_iterations'],
        max_time=params['max_time'],
        depth_limit=params['depth_limit'],
        discount_factor=params['discount_factor'],
        exploration_const=params['exploration_const'],
        sensor_alpha=params['real_alpha'],
        sensor_beta=params['real_beta'],
        reward_alpha=params['reward_alpha'],
        map_size=params['map_size']
    )

    # Variabili loop
    clock = pygame.time.Clock()
    running = True
    steps_taken = 0
    auto_mode = False
    auto_timer = 0
    AUTO_INTERVAL = 1500  # ms

    background_surface = pygame.Surface((GRID_WIDTH, screen_h))
    force_redraw = True

    is_recording = False
    frames = []

    pomcp_stats = {
        'simulations': 0,
        'nodes': 0,
        'time': 0.0,
        'depth': 0,
        'top_actions': [],
        'best_action': ('Stay', 'Stay'),
        'best_q': 0.0
    }

    while running:
        # Auto-mode POMCP
        if auto_mode:
            current_time = pygame.time.get_ticks()
            if current_time - auto_timer > AUTO_INTERVAL:
                # POMCP search
                start_time = time.time()
                best_action = solver.search(belief_map.copy(), drone_positions)
                elapsed = time.time() - start_time

                # Statistiche REALI dal solver
                pomcp_stats['simulations'] = solver.root.visits
                pomcp_stats['nodes'] = solver.total_nodes_created
                pomcp_stats['time'] = elapsed
                pomcp_stats['depth'] = solver.max_depth_reached
                pomcp_stats['top_actions'] = solver.last_top_actions
                pomcp_stats['best_action'] = best_action
                # Estrai Q-value dell'azione migliore dal solver
                pomcp_stats['best_q'] = solver.root.value_estimates.get(best_action, 0.0)

                print(f"\nStep {steps_taken}: Azione scelta {best_action}")

                # Esegui movimento
                drone_positions = execute_move(drone_positions, best_action)

                # Osservazione reale
                obs = get_real_observation(drone_positions, target_pos, params['real_alpha'], params['real_beta'])
                print(f"   Osservazione: {obs}")

                # Update belief
                belief_map = solver.get_updated_belief_map(
                    belief_map,
                    drone_positions[0],
                    drone_positions[1],
                    obs
                )

                steps_taken += 1
                force_redraw = True
                auto_timer = current_time

                # Check terminazione
                max_prob = belief_map.max()
                if max_prob >= 0.95:
                    print("\n!!! TARGET TROVATO !!!")
                    auto_mode = False

        # Eventi
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
                        print("Modalità POMCP AUTO attivata")
                        auto_timer = pygame.time.get_ticks()
                    else:
                        print("Modalità AUTO disattivata")

                if event.key == pygame.K_g:
                    is_recording = not is_recording
                    if is_recording:
                        print("🔴 Registrazione GIF avviata")
                        frames = []
                    else:
                        print("💾 Salvataggio GIF...")
                        filename = f'pomcp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
                        imageio.mimsave(filename, frames, fps=30, loop=0)
                        print(f"✅ GIF salvata: {filename}")
                        frames = []

        # Rendering
        if force_redraw:
            draw_static_background(background_surface, belief_map, font_cell, params)
            force_redraw = False

        # Riempi l'intera finestra di bianco prima di disegnare
        screen.fill((255, 255, 255))
        screen.blit(background_surface, (0, 0))
        draw_elements(
            screen, belief_map, drone_positions, params,
            font_sidebar, GRID_WIDTH, cell_size,
            steps_taken, pomcp_stats, sidebar_w
        )

        if is_recording:
            pygame.draw.circle(screen, (255, 0, 0), (screen_w - 20, 20), 10)

        pygame.display.flip()

        if is_recording:
            rect = pygame.Rect(0, 0, screen_w, screen_h)
            sub = screen.subsurface(rect)
            frame_data = pygame.surfarray.array3d(sub)
            frame_data = np.rot90(frame_data)
            frame_data = np.flipud(frame_data)
            frames.append(frame_data)

        clock.tick(30)


def main():
    while True:
        params = get_user_parameters()
        result = run_simulation(params)
        pygame.quit()
        if result == "quit":
            print("Simulazione terminata.")
            break
        elif result == "restart":
            print("Riavvio...")
            continue

if __name__ == "__main__":
    main()
