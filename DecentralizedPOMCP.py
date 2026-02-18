import math
import random
import time
import numpy as np
from scipy.stats import multivariate_normal
import pygame
import imageio
from datetime import datetime
import multiprocessing


# =============================================================================
# 1. CONFIGURAZIONE E PARAMETRI UTENTE
# =============================================================================

DEFAULT_CONFIG = {
    'map_size': 20,
    'real_alpha': 0.01,
    'real_beta': 0.01,
    'max_iterations': 1000000,
    'max_time': 3.5,
    'depth_limit': 100,
    'discount_factor': 0.9,
    'exploration_const': math.sqrt(2),
    'reward_alpha': 3,
    'penalty_w': 1,
    'penalty_coeff': 0.1,
}

def get_user_parameters():
    """
    Interfaccia grafica interattiva per configurare la missione.
    L'utente clicca sulla griglia per selezionare coordinate, preme INVIO per confermare.
    """
    print("=== CONFIGURAZIONE MISSIONE DI RICERCA (GUI) ===")
    
    cfg = DEFAULT_CONFIG
    map_size = cfg['map_size']
    real_alpha = cfg['real_alpha']
    real_beta = cfg['real_beta']
    print(f"Parametri: Map Size={map_size}x{map_size}, Alpha={real_alpha}, Beta={real_beta}")
    
    # Inizializza pygame
    pygame.init()
    
    # Dimensioni della finestra
    cell_size = 30
    margin = 50
    info_height = 150
    window_width = map_size * cell_size + 2 * margin
    window_height = map_size * cell_size + 2 * margin + info_height
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Configurazione Missione - Click per selezionare")
    
    # Colori
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    LIGHT_GRAY = (240, 240, 240)
    BLUE = (50, 100, 255)
    RED = (255, 50, 50)
    GREEN = (50, 255, 50)
    YELLOW = (255, 200, 50)
    PURPLE = (200, 50, 255)
    
    # Font
    font_title = pygame.font.Font(None, 28)
    font_info = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 18)
    
    def draw_grid(selected_positions, phase_name, phase_color, context_items=None):
        """Disegna la griglia con le posizioni selezionate e gli elementi contestuali
        
        Args:
            selected_positions: lista delle posizioni selezionate nella fase corrente
            phase_name: nome della fase corrente
            phase_color: colore per la fase corrente
            context_items: dizionario con elementi da mostrare come contesto
                          {'drones': [], 'target': None, 'peaks': []}
        """
        screen.fill(WHITE)
        
        if context_items is None:
            context_items = {}
        
        # Titolo
        title_text = font_title.render(f"FASE: {phase_name}", True, phase_color)
        screen.blit(title_text, (margin, 10))
        
        # Istruzioni
        info_text = font_info.render("Click sulla griglia per selezionare celle", True, BLACK)
        help_text = font_small.render("Premi INVIO per confermare e passare alla fase successiva", True, GRAY)
        screen.blit(info_text, (margin, 35))
        screen.blit(help_text, (margin, 58))
        
        # Griglia
        grid_offset_y = margin + info_height
        for r in range(map_size):
            for c in range(map_size):
                x = margin + c * cell_size
                y = grid_offset_y + r * cell_size
                
                # Determina il colore della cella e l'etichetta
                cell_color = LIGHT_GRAY
                label = f"{r},{c}"
                text_color = BLACK
                
                # Controlla se è un drone già inserito
                if 'drones' in context_items and (r, c) in context_items['drones']:
                    cell_color = BLUE
                    label = "D"
                    text_color = WHITE
                # Controlla se è il target già inserito
                elif 'target' in context_items and (r, c) == context_items['target']:
                    cell_color = RED
                    label = "T"
                    text_color = WHITE
                # Controlla se è un picco gaussiano già inserito
                elif 'peaks' in context_items and (r, c) in context_items['peaks']:
                    cell_color = PURPLE
                    label = "G"
                    text_color = WHITE
                # Controlla se è selezionato nella fase corrente
                elif (r, c) in selected_positions:
                    cell_color = phase_color
                    label = f"{r},{c}"
                    text_color = WHITE
                
                pygame.draw.rect(screen, cell_color, (x, y, cell_size, cell_size))
                pygame.draw.rect(screen, GRAY, (x, y, cell_size, cell_size), 1)
                
                # Mostra coordinata o etichetta
                coord_text = font_small.render(label, True, text_color)
                text_rect = coord_text.get_rect(center=(x + cell_size//2, y + cell_size//2))
                screen.blit(coord_text, text_rect)
        
        # Contatore selezioni
        count_text = font_info.render(f"Selezionati: {len(selected_positions)}", True, BLACK)
        screen.blit(count_text, (margin, window_height - 35))
        
        pygame.display.flip()
    
    def get_cell_from_mouse(mouse_pos):
        """Converte posizione mouse in coordinate griglia"""
        grid_offset_y = margin + info_height
        x, y = mouse_pos
        
        if x < margin or y < grid_offset_y:
            return None
        
        c = (x - margin) // cell_size
        r = (y - grid_offset_y) // cell_size
        
        if 0 <= r < map_size and 0 <= c < map_size:
            return (r, c)
        return None
    
    def interactive_selection(phase_name, phase_color, allow_multiple=True, context_items=None):
        """
        Modalità interattiva per selezionare celle.
        
        Args:
            phase_name: nome della fase
            phase_color: colore per la fase corrente
            allow_multiple: consente selezioni multiple
            context_items: dizionario con elementi da mostrare come contesto
        
        Returns: lista di coordinate selezionate
        """
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
                            selected.remove(cell)  # Deseleziona se già selezionato
                        else:
                            if allow_multiple:
                                selected.append(cell)
                            else:
                                selected = [cell]  # Solo una selezione
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if len(selected) > 0:
                            running = False
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()
        
        return selected
    
    # ========== FASE 1: SELEZIONE DRONI ==========
    print("\n[1/5] Seleziona posizioni DRONI (click multipli + INVIO)")
    drone_positions = interactive_selection("1. Posizioni DRONI", BLUE, allow_multiple=True)
    num_drones = len(drone_positions)
    print(f"✓ {num_drones} droni configurati")
    
    # ========== FASE 2: SELEZIONE TARGET ==========
    print("\n[2/5] Seleziona posizione TARGET (1 click + INVIO)")
    # Mostra i droni già inseriti come contesto
    context_with_drones = {'drones': drone_positions}
    target_list = interactive_selection("2. Posizione TARGET", RED, allow_multiple=False, context_items=context_with_drones)
    target_pos = target_list[0]
    print(f"✓ Target in posizione {target_pos}")
    
    # ========== FASE 3: SCELTA TIPO MAPPA ==========
    print("\n[3/5] Scegli tipo di mappa belief iniziale")
    map_config = {}
    
    # Finestra di scelta tipo mappa
    screen.fill(WHITE)
    title = font_title.render("Tipo di Belief Map Iniziale", True, BLACK)
    screen.blit(title, (margin, margin))
    
    options = [
        "1 - Uniforme (premi 1)",
        "2 - Singola Gaussiana (premi 2)",
        "3 - Multi-Gaussiana (premi 3)"
    ]
    
    for i, opt in enumerate(options):
        text = font_info.render(opt, True, BLACK)
        screen.blit(text, (margin, margin + 50 + i * 30))
    
    pygame.display.flip()
    
    map_type = None
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    map_type = 1
                    waiting = False
                elif event.key == pygame.K_2:
                    map_type = 2
                    waiting = False
                elif event.key == pygame.K_3:
                    map_type = 3
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
    
    map_config['type'] = map_type
    map_config['peaks'] = []
    print(f"✓ Tipo mappa: {['Uniforme', 'Singola Gaussiana', 'Multi-Gaussiana'][map_type-1]}")
    
    # ========== FASE 4: SELEZIONE PICCHI GAUSSIANI (se necessario) ==========
    if map_type in [2, 3]:
        # Mostra droni e target già inseriti come contesto
        context_with_drones_target = {'drones': drone_positions, 'target': target_pos}
        
        if map_type == 2:
            print("\n[4/5] Seleziona centro della GAUSSIANA (1 click + INVIO)")
            peak_centers = interactive_selection("4. Centro GAUSSIANA", PURPLE, allow_multiple=False, context_items=context_with_drones_target)
        else:
            print("\n[4/5] Seleziona centri delle GAUSSIANE (click multipli + INVIO)")
            peak_centers = interactive_selection("4. Centri GAUSSIANE", PURPLE, allow_multiple=True, context_items=context_with_drones_target)
        
        # Per ogni picco, chiedi sigma (via terminale per semplicità)
        for i, center in enumerate(peak_centers):
            print(f"\n  Picco #{i+1} in posizione {center}")
            while True:
                try:
                    sigmas = input("  Inserisci Sigma_X, Sigma_Y (es: 2.0,2.0): ")
                    sx, sy = map(float, sigmas.split(','))
                    if sx > 0 and sy > 0:
                        break
                    print("  Le deviazioni standard devono essere positive.")
                except ValueError:
                    print("  Formato errato. Usa: numero,numero")
            
            map_config['peaks'].append({
                'mean': center,
                'cov': [sx, sy]
            })
        
        print(f"✓ {len(peak_centers)} picchi gaussiani configurati")
    else:
        print("\n[4/5] Nessun picco gaussiano necessario (mappa uniforme)")
    
    # ========== FASE 5: SELEZIONE OSTACOLI ==========
    print("\n[5/5] Seleziona posizioni OSTACOLI (click multipli + INVIO, o solo INVIO per nessuno)")
    
    # Estrai i centri delle gaussiane per visualizzarli
    gaussian_centers = [peak['mean'] for peak in map_config['peaks']]
    
    # Mostra la griglia con droni, target e gaussiane già posizionati per riferimento
    def draw_grid_with_context(obstacles):
        """Disegna la griglia mostrando droni, target, gaussiane e ostacoli"""
        screen.fill(WHITE)
        
        title_text = font_title.render("FASE: 5. Posizioni OSTACOLI", True, BLACK)
        screen.blit(title_text, (margin, 10))
        
        info_text = font_info.render("Click per aggiungere ostacoli (evita droni e target!)", True, BLACK)
        help_text = font_small.render("Premi INVIO per confermare (anche senza ostacoli)", True, GRAY)
        screen.blit(info_text, (margin, 35))
        screen.blit(help_text, (margin, 58))
        
        grid_offset_y = margin + info_height
        for r in range(map_size):
            for c in range(map_size):
                x = margin + c * cell_size
                y = grid_offset_y + r * cell_size
                
                cell_color = LIGHT_GRAY
                label = f"{r},{c}"
                text_color = BLACK
                
                # Droni
                if (r, c) in drone_positions:
                    cell_color = BLUE
                    label = "D"
                    text_color = WHITE
                # Target
                elif (r, c) == target_pos:
                    cell_color = RED
                    label = "T"
                    text_color = WHITE
                # Centri Gaussiane
                elif (r, c) in gaussian_centers:
                    cell_color = PURPLE
                    label = "G"
                    text_color = WHITE
                # Ostacoli
                elif (r, c) in obstacles:
                    cell_color = BLACK
                    label = "X"
                    text_color = WHITE
                
                pygame.draw.rect(screen, cell_color, (x, y, cell_size, cell_size))
                pygame.draw.rect(screen, GRAY, (x, y, cell_size, cell_size), 1)
                
                coord_text = font_small.render(label, True, text_color)
                text_rect = coord_text.get_rect(center=(x + cell_size//2, y + cell_size//2))
                screen.blit(coord_text, text_rect)
        
        count_text = font_info.render(f"Ostacoli: {len(obstacles)}", True, BLACK)
        screen.blit(count_text, (margin, window_height - 35))
        
        pygame.display.flip()
    
    obstacles = []
    running = True
    
    while running:
        draw_grid_with_context(obstacles)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                cell = get_cell_from_mouse(event.pos)
                if cell:
                    # Non permettere ostacoli su droni o target
                    if cell in drone_positions or cell == target_pos:
                        print(f"  ⚠ Non puoi mettere un ostacolo su drone/target!")
                        continue
                    
                    if cell in obstacles:
                        obstacles.remove(cell)
                    else:
                        obstacles.append(cell)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    running = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
    
    print(f"✓ {len(obstacles)} ostacoli configurati")
    
    # Chiudi la finestra pygame
    pygame.quit()
    
    print("\n=== CONFIGURAZIONE COMPLETATA ===")
    print(f"Droni: {num_drones}")
    print(f"Target: {target_pos}")
    print(f"Tipo mappa: {['Uniforme', 'Singola Gaussiana', 'Multi-Gaussiana'][map_type-1]}")
    print(f"Ostacoli: {len(obstacles)}")
    
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
        'num_drones': num_drones,
        'drone_positions': drone_positions,
        'target_pos': target_pos,
        'map_config': map_config,
        'obstacles': obstacles
    }

# Funzione per generazione della mappa degli ostacoli
def initialize_obstacle_map(params):
    """
    Crea una mappa booleana per gli ostacoli.
    1 = cella con ostacolo
    0 = cella libera
    """
    map_size = params['map_size']
    obstacle_map = np.zeros((map_size, map_size), dtype=int)
    
    for obs_pos in params['obstacles']:
        r, c = obs_pos
        obstacle_map[r, c] = 1
    
    return obstacle_map

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

    # Applicazione ostacoli PRIMA della normalizzazione
    # Se è presente un ostacolo, la probabilità cade a zero
    if 'obstacles' in params and len(params['obstacles']) > 0:
        obstacle_map = initialize_obstacle_map(params)
        # Moltiplica ogni cella per (1 - presenza_ostacolo)
        # Se presenza_ostacolo = 1, la probabilità diventa 0
        # Se presenza_ostacolo = 0, la probabilità resta invariata
        belief_map = belief_map * (1 - obstacle_map)
    
    # Normalizzazione della mappa + check somma di tutte le celle deve fare 1.0 
    total_prob = np.sum(belief_map)
    
    if total_prob == 0:
        # Fallback di sicurezza per valori di varianze enormi o troppi ostacoli
        belief_map.fill(1.0 / (map_size * map_size))
    else:
        belief_map /= total_prob
        
    return belief_map


# =============================================================================
# 2. POMCP LOGIC (FEDELTÀ 100% ORIGINALE)
# =============================================================================

# Costante globale per movimenti (evita duplicazione)
MOVES_DELTA = {
    'N': (-1, 0),
    'S': (1, 0),
    'W': (0, -1),
    'E': (0, 1),
    'Stay': (0, 0)
}

def resolve_conflicts_centralized(drones, intention_packets):
    """
    Risolve i conflitti tra droni con logica sequenziale e coerente.
    
    ALGORITMO:
    1. Ordina tutti i droni per best_q decrescente
    2. Assegna best_action in ordine, controllando conflitti con azioni già decise
    3. Se best_action crea conflitto → drone va nel set dei "perdenti"
    4. Ordina i perdenti per second_q decrescente
    5. Assegna second_action in ordine, controllando conflitti con tutte le azioni già decise
    6. Se anche second_action crea conflitto → Stay
    
    Args:
        drones: lista di oggetti DroneAgent
        intention_packets: lista di pacchetti con best_action, best_q, second_action, second_q
    
    Returns:
        dict: {drone_id: final_action}
    """
    # Dizionari per accesso rapido
    pkt_by_id = {pkt['id']: pkt for pkt in intention_packets}
    
    # Risultati finali
    final_actions = {}  # {drone_id: action}
    final_positions = {}  # {drone_id: (r, c)}
    
    # ========== FASE 1: PROCESSAMENTO BEST_ACTION ==========
    # Ordina tutti i droni per best_q decrescente (a parità, ID crescente)
    drones_by_best_q = sorted(pkt_by_id.keys(), 
                              key=lambda did: (pkt_by_id[did]['best_q'], -did), 
                              reverse=True)
    
    print(f"\n[PHASE 1] Processamento best_action in ordine di best_q:")
    print(f"  Posizioni correnti droni:")
    for did in sorted(pkt_by_id.keys()):
        pkt = pkt_by_id[did]
        print(f"    D{did}: pos={pkt['pos']}, best={pkt['best_action']} (Q={pkt['best_q']:.2f}), second={pkt['second_action']} (Q={pkt['second_q']:.2f})")
    
    # Set dei droni che non riescono ad eseguire best_action
    rejected_drones = []
    
    for drone_id in drones_by_best_q:
        pkt = pkt_by_id[drone_id]
        action = pkt['best_action']
        delta = MOVES_DELTA[action]
        next_pos = (pkt['pos'][0] + delta[0], pkt['pos'][1] + delta[1])
        
        print(f"  D{drone_id}: valuto best_action={action}, next_pos={next_pos}")
        
        # Controlla se questa best_action crea conflitto con azioni già decise
        has_conflict = False
        conflict_with = None
        
        for other_id, other_pos in final_positions.items():
            other_pkt = pkt_by_id[other_id]
            
            # Conflitto: collisione (stessa cella target)
            if next_pos == other_pos:
                has_conflict = True
                conflict_with = other_id
                break
            
            # Conflitto: swap (scambio di posizioni)
            if next_pos == other_pkt['pos'] and other_pos == pkt['pos']:
                has_conflict = True
                conflict_with = other_id
                break
        
        if has_conflict:
            # Best_action in conflitto → va nel set dei perdenti
            rejected_drones.append(drone_id)
            print(f"  D{drone_id}: best={action} (Q={pkt['best_q']:.2f}) → CONFLITTO con D{conflict_with} → passa a second_action")
        else:
            # Nessun conflitto → assegna best_action
            final_actions[drone_id] = action
            final_positions[drone_id] = next_pos
            print(f"  D{drone_id}: best={action} (Q={pkt['best_q']:.2f}) → OK")
    
    # ========== FASE 2: PROCESSAMENTO SECOND_ACTION ==========
    if rejected_drones:
        print(f"\n[PHASE 2] Processamento second_action per {len(rejected_drones)} droni in ordine di second_q:")
        print(f"  Posizioni già assegnate in final_positions: {final_positions}")
        
        # Ordina i droni rifiutati per second_q decrescente (a parità, ID crescente)
        rejected_sorted = sorted(rejected_drones,
                               key=lambda did: (pkt_by_id[did]['second_q'], -did),
                               reverse=True)
        
        for drone_id in rejected_sorted:
            pkt = pkt_by_id[drone_id]
            action = pkt['second_action']
            delta = MOVES_DELTA[action]
            next_pos = (pkt['pos'][0] + delta[0], pkt['pos'][1] + delta[1])
            
            # Controlla se questa second_action crea conflitto con TUTTE le azioni già decise
            has_conflict = False
            conflict_with = None
            
            for other_id, other_pos in final_positions.items():
                other_pkt = pkt_by_id[other_id]
                
                # Conflitto: collisione
                if next_pos == other_pos:
                    has_conflict = True
                    conflict_with = other_id
                    break
                
                # Conflitto: swap
                if next_pos == other_pkt['pos'] and other_pos == pkt['pos']:
                    has_conflict = True
                    conflict_with = other_id
                    break
            
            if has_conflict:
                # Anche second_action in conflitto → Stay
                final_actions[drone_id] = 'Stay'
                final_positions[drone_id] = pkt['pos']
                print(f"  D{drone_id}: second={action} (Q={pkt['second_q']:.2f}) → CONFLITTO con D{conflict_with} → Stay")
            else:
                # Nessun conflitto → assegna second_action
                final_actions[drone_id] = action
                final_positions[drone_id] = next_pos
                if action == 'Stay':
                    print(f"  D{drone_id}: second=Stay (Q={pkt['second_q']:.2f}) → OK (scelta POMCP)")
                else:
                    print(f"  D{drone_id}: second={action} (Q={pkt['second_q']:.2f}) → OK")
    
    # ========== STATISTICHE FINALI ==========
    print(f"\n[SUMMARY] Azioni finali decise:")
    for drone_id in sorted(final_actions.keys()):
        pkt = pkt_by_id[drone_id]
        final_act = final_actions[drone_id]
        status = ""
        if final_act == pkt['best_action']:
            status = "✓ best_action"
        elif final_act == pkt['second_action']:
            status = "→ second_action"
        else:  # Stay forzato
            status = "✗ Stay (conflitto)"
        print(f"  D{drone_id}: {final_act} {status}")
    
    stay_count = sum(1 for act in final_actions.values() if act == 'Stay')
    stay_forced = sum(1 for did, act in final_actions.items() 
                     if act == 'Stay' and act != pkt_by_id[did]['best_action'] and act != pkt_by_id[did]['second_action'])
    
    if stay_forced > 0:
        print(f"\n[WARNING] {stay_forced}/{len(final_actions)} droni forzati a Stay per conflitti irrisolvibili")
    
    return final_actions

def precompute_all_pairs_distances(map_size, obstacle_map):
    """
    Pre-calcola TUTTE le distanze tra coppie di celle libere usando BFS.
    Tiene conto dei muri (obstacle_map) per calcolare la distanza reale.
    
    Args:
        map_size: Dimensione della mappa (NxN)
        obstacle_map: Matrice NxN dove 1 = ostacolo, 0 = libero
    
    Returns:
        dist_lookup: Dizionario {(start_pos, end_pos): distanza}
                     Se la coppia non esiste, significa che end_pos è irraggiungibile da start_pos
    """
    from collections import deque
    
    dist_lookup = {}
    
    # Direzioni di movimento (escluso 'Stay' perché cerchiamo il percorso più breve)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
    
    # Itera su ogni cella libera come punto di partenza
    for start_r in range(map_size):
        for start_c in range(map_size):
            start_pos = (start_r, start_c)
            
            # Salta ostacoli
            if obstacle_map[start_r, start_c] == 1:
                continue
            
            # BFS da questa cella verso tutte le altre
            queue = deque([(start_pos, 0)])  # (posizione, distanza)
            visited = {start_pos}
            
            while queue:
                current_pos, dist = queue.popleft()
                
                # Salva la distanza nella lookup table
                dist_lookup[(start_pos, current_pos)] = dist
                
                # Esplora i vicini
                for dr, dc in directions:
                    next_r = current_pos[0] + dr
                    next_c = current_pos[1] + dc
                    next_pos = (next_r, next_c)
                    
                    # Verifica limiti della mappa
                    if not (0 <= next_r < map_size and 0 <= next_c < map_size):
                        continue
                    
                    # Salta ostacoli
                    if obstacle_map[next_r, next_c] == 1:
                        continue
                    
                    # Salta se già visitato
                    if next_pos in visited:
                        continue
                    
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))
    
    return dist_lookup

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

class POMCPSolver:
    def __init__(self, max_iterations=None, max_time=None, depth_limit=None, discount_factor=None,
                 exploration_const=None, sensor_alpha=None, sensor_beta=None,
                 reward_alpha=None, map_size=None, obstacle_map=None,
                 penalty_w=None, penalty_coeff=None, drone_id=None):
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
        self.obstacle_map = obstacle_map if obstacle_map is not None else np.zeros((self.map_size, self.map_size), dtype=int)

        self.total_nodes_created = 0  # Contatore nodi creati durante search
        self.max_depth_reached = 0    # Profondità massima raggiunta
        
        # Parametri per nuova formula penalty: total_reward = base_reward - penalty_coeff * penalty
        self.penalty_w = penalty_w if penalty_w is not None else cfg['penalty_w']
        self.penalty_coeff = penalty_coeff if penalty_coeff is not None else cfg['penalty_coeff']
        
        # ID del drone per gestire priorità asimmetrica nella penalty
        self.drone_id = drone_id
        
        # PRE-CALCOLO LOOKUP TABLE: Calcola TUTTE le distanze tra coppie di celle all'avvio
        # Questo viene fatto UNA SOLA VOLTA e poi riutilizzato per tutti i rollout
        self.dist_lookup = precompute_all_pairs_distances(self.map_size, self.obstacle_map)

    # Funzione di ricerca POMCP: costruiamo albero + restituzione azione migliore finale
    def search(self, current_belief_map, drone_position, partner_positions=None, partner_plans=None):
        
        # Creazione del nodo radice con la belief map attuale
        root = POMCPNode(belief_map=current_belief_map)
        self.root = root  
        self.total_nodes_created = 1  
        self.max_depth_reached = 0
        
        # PRIORITÀ ASIMMETRICA: crea lista droni da ignorare nella penalty
        # Ignoro droni con ID > mio ID e distanza Manhattan <= 2
        ignored_penalty_drones = set()
        if self.drone_id is not None and partner_positions is not None:
            for partner_id, partner_pos in partner_positions.items():
                # Calcola distanza Manhattan tra la mia posizione e quella del partner
                dist_manhattan = abs(drone_position[0] - partner_pos[0]) + abs(drone_position[1] - partner_pos[1])
                
                # Se il partner ha ID superiore al mio E distanza <= 2, lo ignoro nella penalty
                if partner_id > self.drone_id and dist_manhattan <= 2:
                    ignored_penalty_drones.add(partner_id)
        
        start_time = time.time()
        
        # Ciclo principale di simulazione Monte Carlo
        for i in range(self.max_iterations):
            if (time.time() - start_time) > self.max_time:
                break
            
            # Campionamento stato iniziale: estrazione posizione target ad ogni iterazione (stato: pos droni e pos target)
            sampled_target_pos = self._sample_target_from_belief(root.belief_map)
            state = (sampled_target_pos, drone_position)
            
            # Avvio della simulazione ricorsiva 
            self.simulate(state, root, 0, partner_positions=partner_positions, partner_plans=partner_plans, 
                         ignored_penalty_drones=ignored_penalty_drones)
        
        # Selezione delle top 2 azioni migliori
        best_action, best_q, second_action, second_q = self._select_top_two_actions(root)
        
        # Estrazione dei piani futuri per tutte le azioni dalla radice
        future_plans = self._extract_future_plans(root, drone_position)
        
        return best_action, best_q, second_action, second_q, future_plans

    # Singola simulazione POMCP: fatta in maniera ricorsiva per scendere in profondità
    def simulate(self, state, node, depth, visited_cells=None, partner_positions=None, partner_plans=None, 
                 ignored_penalty_drones=None): 
        
        # Inizializzazione del set alla radice (COPIA per evitare condivisione)
        if visited_cells is None:
            visited_cells = set()
        else:
            # Crea una copia per questo ramo dell'albero
            visited_cells = visited_cells.copy()

        # Aggiorna profondità massima raggiunta finora
        if depth > self.max_depth_reached:
            self.max_depth_reached = depth

        # Controllo terminazione (Depth o Stato Terminale: se target trovato in simulazione)
        if depth >= self.depth_limit:
            return 0.0

        # Espansione e Rollout 
        if node.is_leaf():
            # Se il nodo non ha figli, generiamo le azioni possibili
            # FASE 2: Al root level, escludiamo posizione partner
            is_root = (node.parent is None)
            self.expand(node, state, partner_positions if is_root else None)
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
        next_state, observation, reward, terminal = self.generative_model_G(
            state, action, node.belief_map, visited_cells, depth, partner_positions, partner_plans,
            ignored_penalty_drones
        )

        # Discesa nell'albero: verifica se esiste nodo figlio 
        if (action, observation) in node.children:
            child_node = node.children[(action, observation)]
        else:
            
            # Estrazione nuova posizione dal next_state per aggiornare la mappa
            _, next_drone_pos = next_state
            # Calcoliamo la nuova belief map
            new_belief_map = self.get_updated_belief_map(node.belief_map, next_drone_pos, observation)
            
            child_node = POMCPNode(belief_map=new_belief_map, parent=node)
            node.children[(action, observation)] = child_node
            self.total_nodes_created += 1  

        # Ricorsione o Stop se Terminale 
        if terminal:
            future_reward = 0.0
        else:
            future_reward = self.simulate(next_state, child_node, depth + 1, visited_cells, partner_positions=None, partner_plans=partner_plans, ignored_penalty_drones=ignored_penalty_drones)
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
    def expand(self, node, state, partner_positions=None):
        
        # Estrazione posizione attuale del drone dallo stato
        _, drone_pos = state
        
        map_size = self.map_size  # Dimensione della griglia
        
        # Estrai lista posizioni partner da dizionario {drone_id: position}
        # partner_positions è None nei livelli > root, altrimenti dizionario
        partner_list = []
        if partner_positions is not None:
            # Dizionario {drone_id: position} -> estrai solo le posizioni
            partner_list = list(partner_positions.values())

        for action in MOVES_DELTA.keys():

            delta = MOVES_DELTA[action]
            next_pos = (drone_pos[0] + delta[0], drone_pos[1] + delta[1])

            # Verifica confini mappa
            if not (0 <= next_pos[0] < map_size and 0 <= next_pos[1] < map_size):
                continue

            # VIETARE celle con ostacoli
            if self.obstacle_map[next_pos[0], next_pos[1]] == 1:
                continue

            # VIETARE celle occupate da partner (solo al root, depth=0)
            # Previene collisioni immediate con posizioni attuali degli altri droni
            if partner_list and next_pos in partner_list:
                continue

            if action not in node.action_counts:
                node.action_counts[action] = 0
                node.value_estimates[action] = 0.0
        
        # Safety: garantiamo che Stay sia sempre disponibile come fallback
        if not node.action_counts:
            node.action_counts['Stay'] = 0
            node.value_estimates['Stay'] = 0.0

    # Rollout leggero basato su euristica di distanza REALE (con lookup table)
    def rollout(self, state):
        
        target_pos, drone_pos = state
        
        # OTTIMIZZAZIONE: Lettura O(1) dalla lookup table invece di calcolo
        # La distanza tiene conto dei muri (distanza reale, non Manhattan)
        dist = self.dist_lookup.get((drone_pos, target_pos))
        
        # Se la coppia non esiste nella lookup, il target è irraggiungibile
        # Restituiamo una reward molto bassa (distanza infinita)
        if dist is None:
            return 0.0  # Reward minima per target irraggiungibile
        
        # Reward decrescente con la distanza REALE
        score = 1 * (self.gamma ** dist)
        return score

    #Black box simulator: transizione di stato(movimento droni), osservazione, reward
    def generative_model_G(self, state, action, belief_map, visited_cells, depth=0, 
                          partner_positions=None, partner_plans=None, ignored_penalty_drones=None):
        
        target_pos, drone_pos = state
        
        # 1. Transizione di Stato (Deterministica)
        delta = MOVES_DELTA[action]
        next_drone = (drone_pos[0] + delta[0], drone_pos[1] + delta[1])
        next_state = (target_pos, next_drone)

        # 2. Generazione dell'osservazione per il singolo drone
        is_target_present = (next_drone == target_pos)
        if is_target_present:
            obs = 0 if np.random.rand() < self.sensor_beta else 1
        else:
            obs = 1 if np.random.rand() < self.sensor_alpha else 0

        # 3. Calcolo della Reward Base
        # Formula: R_base = R_target + reward_alpha * R_token                
        terminal = False
        if is_target_present and obs == 1:
            r_target = 1.0
            terminal = True
        else:
            r_target = 0.0

        r_token = 0.0
        if next_drone not in visited_cells:
            if 0 <= next_drone[0] < self.map_size and 0 <= next_drone[1] < self.map_size:
                r_token += belief_map[next_drone]
            visited_cells.add(next_drone)

        base_reward = r_target + (self.reward_alpha * r_token)
        
        # 4. Calcolo Penalty Repulsiva (Nuova Formula)
        # Formula: penalty = (w/2) * SUM{1/(d^2)}
        # dove d^2 è la distanza di Manhattan al quadrato
        penalty = 0.0
        
        # ALLINEAMENTO TEMPORALE depth vs future_plan:
        # - depth=0: io eseguo la mia prima azione, partner esegue future_plan[0]
        # - depth=1: io eseguo la mia seconda azione, partner esegue future_plan[1]
        # - depth=n: io eseguo la mia (n+1)-esima azione, partner esegue future_plan[n]
        #
        # Per calcolare la POSIZIONE del partner dopo future_plan[n],
        # parto dalla sua posizione iniziale (partner_positions) e applico
        # TUTTE le azioni da future_plan[0] a future_plan[n]
        if partner_plans is not None and partner_positions is not None:
            for partner_id, future_plan in partner_plans.items():
                # PRIORITÀ ASIMMETRICA: Ignora penalty per droni con ID superiore e vicini
                if ignored_penalty_drones is not None and partner_id in ignored_penalty_drones:
                    continue  # Skippa questo drone, non contribuisce alla penalty
                
                # Verifica: il piano ha l'azione per questo livello di depth?
                # Se non è specificata la posizione futura (depth >= len(future_plan)),
                # non applicare la penalty
                if depth < len(future_plan):
                    # Calcolo posizione del partner dopo aver eseguito future_plan[depth]
                    partner_pos = partner_positions[partner_id]
                    
                    # Applico TUTTE le azioni da future_plan[0] a future_plan[depth]
                    # per ottenere la posizione finale del partner
                    # Esempi:
                    #   depth=0: applico solo future_plan[0] → 1 azione
                    #   depth=1: applico future_plan[0] e [1] → 2 azioni
                    #   depth=2: applico future_plan[0], [1], [2] → 3 azioni
                    for i in range(depth + 1):
                        action_str = future_plan[i]
                        delta_partner = MOVES_DELTA.get(action_str, (0, 0))
                        partner_pos = (partner_pos[0] + delta_partner[0], 
                                      partner_pos[1] + delta_partner[1])
                    
                    # Calcola distanza di MANHATTAN tra next_drone e partner_pos
                    # Formula: d = |x1-x2| + |y1-y2|
                    dist_manhattan = (abs(next_drone[0] - partner_pos[0]) + 
                                     abs(next_drone[1] - partner_pos[1]))
                    
                    # Calcola d^2 (distanza al quadrato)
                    d_squared = dist_manhattan ** 2
                    
                    # Evita divisione per zero: se i droni sono nella stessa posizione
                    if d_squared == 0:
                        # Penalty massima per sovrapposizione
                        penalty += 1e6  # Valore molto alto
                    else:
                        # Calcola penalty per questo partner: 1/(d^2)
                        penalty += 1.0 / d_squared
        
        # Applica il fattore (w/2) alla somma delle penalty
        penalty = (self.penalty_w / 2.0) * penalty
        
        # Reward finale: total_reward = base_reward - coefficiente * penalty
        total_reward = base_reward - self.penalty_coeff * penalty

        return next_state, obs, total_reward, terminal

    # Aggiornamento bayesiano della belief map con singolo sensore
    def get_updated_belief_map(self, current_belief, drone_pos, observation):
        
        return self._single_sensor_update(current_belief, drone_pos, observation)

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
            return belief_map  # Nessuna modifica necessaria

        # 4. Calcolo del nuovo belief map (IMPORTANTE: fare copia per non modificare l'originale)
        new_belief_map = (belief_map.copy() * Phi) / Z

        # Correzione della cella ispezionata 
        new_belief_map[inspected_cell] = (Psi * p_st) / Z
        
        # Applicazione ostacoli: azzera probabilità nelle celle con ostacoli
        new_belief_map = new_belief_map * (1 - self.obstacle_map)
        
        # 5. Normalizzazione esplicita per evitare deriva numerica
        # Rinormalizziamo sulla somma totale delle celle libere
        total = np.sum(new_belief_map)
        if total > 1e-9:  # Protezione contro somma zero
            new_belief_map /= total
        else:
            # Caso estremo: ritorna distribuzione uniforme SOLO sulle celle libere
            free_cells_mask = (1 - self.obstacle_map)
            num_free_cells = np.sum(free_cells_mask)
            if num_free_cells > 0:
                new_belief_map = free_cells_mask / num_free_cells
            else:
                # Caso estremo: nessuna cella libera (non dovrebbe mai accadere)
                new_belief_map = np.ones_like(belief_map) / belief_map.size

        return new_belief_map

    
    # Estrazione posizione target per POMCP
    def _sample_target_from_belief(self, belief_map):
        
        flat_probs = belief_map.flatten()
        
        # Protezione: normalizza se necessario
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

        # Fallback di sicurezza: se best_action è ancora None
        if best_action is None:
            if node.action_counts:
                best_action = random.choice(list(node.action_counts.keys()))
            else:
                # Caso estremo: nessuna azione disponibile
                return 'Stay'

        return best_action

    # Selezione delle top 2 azioni migliori (per gestione conflitti)
    def _select_top_two_actions(self, node):
        
        action_q_pairs = [(action, q_val) for action, q_val in node.value_estimates.items()]
        action_q_pairs.sort(key=lambda x: x[1], reverse=True)
        
        best_action = action_q_pairs[0][0] if len(action_q_pairs) > 0 else 'Stay'
        best_q = action_q_pairs[0][1] if len(action_q_pairs) > 0 else 0.0
        
        second_action = action_q_pairs[1][0] if len(action_q_pairs) > 1 else best_action
        second_q = action_q_pairs[1][1] if len(action_q_pairs) > 1 else best_q
        
        return best_action, best_q, second_action, second_q

    def _extract_future_plans(self, root, drone_position):
        """
        Estrae i piani futuri per ogni azione possibile dalla radice.
        
        Per ogni azione A dalla radice:
        1. Aggiungi A al piano
        2. Trova il figlio più visitato tra quelli con azione A
        3. Dal figlio: scegli l'azione con Q-value più alto
        4. Aggiungi quella azione al piano
        5. Trova il figlio più visitato con quella azione
        6. Ripeti fino a che l'albero ha informazioni disponibili (max 5 azioni)
        
        IMPORTANTE: Se il POMCP non ha raggiunto una depth sufficiente, il piano
        conterrà solo le azioni effettivamente esplorate (senza padding con 'Stay').
        
        Returns:
            Dict[action -> List[actions]]: Mappa azione root -> sequenza azioni future
                                          (lunghezza variabile da 1 a 5 azioni)
        """
        future_plans = {}
        
        # Per ogni azione possibile dalla radice
        for root_action in root.action_counts.keys():
            plan = [root_action]  # Il piano inizia con l'azione root
            
            # Trova tutti i figli che corrispondono a questa azione
            # children key format: (action, observation)
            matching_children = [(action_obs, child) for action_obs, child in root.children.items() 
                                 if action_obs[0] == root_action]
            
            if not matching_children:
                # Nessun figlio trovato: il piano contiene solo l'azione root
                future_plans[root_action] = [root_action]
                continue
            
            # Scegli il figlio più visitato tra quelli che corrispondono all'azione
            most_visited_child = max(matching_children, key=lambda x: x[1].visits)[1]
            current_node = most_visited_child
            
            # Segui il percorso per max 4 livelli aggiuntivi (totale max 5 azioni)
            for depth in range(4):
                # Se il nodo non ha figli o non ha stime di valore, STOP
                if not current_node.value_estimates or current_node.visits == 0:
                    # Albero finito qui: piano contiene solo le azioni raggiunte
                    break
                
                # Filtra solo le azioni effettivamente visitate (N(b,a) > 0)
                # Evita di selezionare azioni espanse ma mai esplorate (Q=0.0, N=0)
                visited_actions = {action: q_val 
                                  for action, q_val in current_node.value_estimates.items() 
                                  if current_node.action_counts.get(action, 0) > 0}
                
                if not visited_actions:
                    # Nessuna azione visitata: STOP
                    break
                
                # Scegli l'azione con il Q-value più alto tra quelle visitate
                best_action_here = max(visited_actions.items(), 
                                      key=lambda x: x[1])[0]
                
                plan.append(best_action_here)
                
                # Trova tutti i figli che corrispondono a questa azione (con Q-value più alto)
                matching_children_next = [(action_obs, child) 
                                         for action_obs, child in current_node.children.items() 
                                         if action_obs[0] == best_action_here]
                
                if not matching_children_next:
                    # Nessun figlio per questa azione: STOP
                    break
                
                # Avanza al figlio più visitato tra quelli con l'azione scelta
                current_node = max(matching_children_next, key=lambda x: x[1].visits)[1]
            
            # Salva il piano con la lunghezza effettivamente raggiunta (1 a 5 azioni)
            future_plans[root_action] = plan
        
        return future_plans


# Worker function per POMCP parallelo
def worker_pomcp_task(args):
    """Worker per eseguire POMCP in modo parallelo"""
    params, belief_map, my_pos, partner_positions, partner_plans, drone_id = args
    
    # Creazione mappa ostacoli
    obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else np.zeros((params['map_size'], params['map_size']), dtype=int)
    
    # Importante: Qui il worker istanzia il solver.
    # Grazie al multiprocessing, questo avviene in uno spazio di memoria separato.
    solver = POMCPSolver(
        max_iterations=params['max_iterations'],
        max_time=params['max_time'],
        depth_limit=params['depth_limit'],
        discount_factor=params['discount_factor'],
        exploration_const=params['exploration_const'],
        sensor_alpha=params['real_alpha'],
        sensor_beta=params['real_beta'],
        reward_alpha=params['reward_alpha'],
        map_size=params['map_size'],
        obstacle_map=obstacle_map,
        penalty_w=params.get('penalty_w'),
        penalty_coeff=params.get('penalty_coeff'),
        drone_id=drone_id
    )
    
    # partner_positions è ora un dizionario {drone_id: position}
    # partner_plans è un dizionario {drone_id: [future_actions]}
    best_action, best_q, second_action, second_q, future_plans = solver.search(
        belief_map, my_pos, partner_positions, partner_plans
    )
    
    return {
        'best_action': best_action,
        'best_q': best_q,
        'second_action': second_action,
        'second_q': second_q,
        'depth': solver.max_depth_reached,
        'visits': solver.root.visits,
        'nodes_created': solver.total_nodes_created,
        'future_plans': future_plans
    }


# =============================================================================
# 3. DRONE AGENT (ENTITÀ DECENTRALIZZATA - NUOVA CLASSE)
# =============================================================================

class DroneAgent:
    """
    Rappresenta un drone autonomo in un sistema decentralizzato.
    Gestisce la propria memoria e comunica via messaggi.
    """
    def __init__(self, drone_id, start_pos, params):
        self.id = drone_id
        self.pos = start_pos
        self.params = params
        
        # MEMORIA PRIVATA: La propria versione della verità (belief map)
        self.belief_map = initialize_belief_map(params)
        
        # Mappa ostacoli (condivisa tra tutti)
        self.obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else np.zeros((params['map_size'], params['map_size']), dtype=int)
        
        # Strumento matematico per update bayesiano locale
        # (Riutilizziamo la logica matematica della classe originale, ma istanziata localmente)
        self.solver_tool = POMCPSolver(
            map_size=params['map_size'], 
            sensor_alpha=params['real_alpha'], 
            sensor_beta=params['real_beta'],
            obstacle_map=self.obstacle_map,
            penalty_w=params.get('penalty_w'),
            penalty_coeff=params.get('penalty_coeff')
        )

        # Stato interno decisionale
        self.planned_result = None   
        self.final_action = None
        
        # Buffer per piani futuri: Dict[action -> List[actions]]
        self.future_plans_buffer = {}
        
        # Memoria dei piani futuri ricevuti dagli altri droni
        # {drone_id: [future_actions]}
        self.partner_future_plans = {}     

    def get_planning_args(self, other_drones):
        """
        PREPARA I DATI per il planner parallelo.
        other_drones: lista di DroneAgent (gli altri droni, escluso self)
        Restituisce una tupla con belief map e info sui partner.
        """
        # Crea dizionario con posizioni attuali dei partner
        partner_positions = {other.id: other.pos for other in other_drones}
        
        # I piani futuri sono già memorizzati in self.partner_future_plans
        # (ricevuti al timestep precedente)
        
        return (self.params, self.belief_map.copy(), self.pos, 
                partner_positions, self.partner_future_plans.copy(), self.id)

    def set_planning_result(self, result):
        """Riceve il risultato dal worker multiprocessing"""
        self.planned_result = result
        
        # Salva i piani futuri nel buffer
        if 'future_plans' in result:
            self.future_plans_buffer = result['future_plans']

    # --- COMUNICAZIONE 1: INTENZIONI ---
    def create_intention_packet(self):
        """Crea un pacchetto con l'intenzione di movimento e il Q-value"""
        if self.planned_result is None:
            # Fallback di sicurezza
            return {
                'id': self.id,
                'pos': self.pos,
                'best_action': 'Stay',
                'best_q': 0.0,
                'second_action': 'Stay',
                'second_q': 0.0
            }
        
        return {
            'id': self.id,
            'pos': self.pos,
            'best_action': self.planned_result['best_action'],
            'best_q': self.planned_result['best_q'],
            'second_action': self.planned_result['second_action'],
            'second_q': self.planned_result['second_q']
        }

    def resolve_conflict_locally(self, other_packets):
        """
        Riceve le intenzioni di tutti gli altri droni e decide l'azione da eseguire.
        
        Logica:
        1. Prova best_action
        2. Se c'è conflitto (stessa cella o swap) con un altro drone:
           - Chi ha Q-value maggiore vince (esegue best_action)
           - Chi ha Q-value minore perde (prova second_action)
        3. Se anche second_action ha conflitti -> Stay
        
        Args:
            other_packets: lista di pacchetti degli altri droni
        
        Returns:
            bool: True se ha dovuto cambiare dalla best_action
        """
        # Safety check
        if self.planned_result is None:
            self.final_action = 'Stay'
            return False
        
        # Prova BEST ACTION
        my_best_act = self.planned_result['best_action']
        my_best_q = self.planned_result['best_q']
        delta = MOVES_DELTA[my_best_act]
        my_next = (self.pos[0] + delta[0], self.pos[1] + delta[1])
        
        # Check conflitti con tutti gli altri droni
        has_conflict_best = False
        for other_pkt in other_packets:
            other_act = other_pkt['best_action']
            other_q = other_pkt['best_q']
            other_delta = MOVES_DELTA[other_act]
            other_next = (other_pkt['pos'][0] + other_delta[0], other_pkt['pos'][1] + other_delta[1])
            
            # Conflitto: stessa cella
            collision = (my_next == other_next)
            # Swap: scambio posizioni
            swap = (my_next == other_pkt['pos'] and other_next == self.pos)
            
            if collision or swap:
                # Chi ha Q più alto vince, a parità ID minore vince
                if other_q > my_best_q or (other_q == my_best_q and other_pkt['id'] < self.id):
                    has_conflict_best = True
                    break
        
        # Se best_action non ha conflitti, usala
        if not has_conflict_best:
            self.final_action = my_best_act
            return False  # Non ho cambiato
        
        # Prova SECOND ACTION
        my_second_act = self.planned_result['second_action']
        my_second_q = self.planned_result['second_q']
        delta2 = MOVES_DELTA[my_second_act]
        my_next2 = (self.pos[0] + delta2[0], self.pos[1] + delta2[1])
        
        # Check conflitti per second_action
        has_conflict_second = False
        for other_pkt in other_packets:
            other_act = other_pkt['best_action']
            other_q = other_pkt['best_q']
            other_delta = MOVES_DELTA[other_act]
            other_next = (other_pkt['pos'][0] + other_delta[0], other_pkt['pos'][1] + other_delta[1])
            
            collision = (my_next2 == other_next)
            swap = (my_next2 == other_pkt['pos'] and other_next == self.pos)
            
            if collision or swap:
                # Confronto second_q con other best_q
                if other_q > my_second_q or (other_q == my_second_q and other_pkt['id'] < self.id):
                    has_conflict_second = True
                    break
        
        # Se second_action non ha conflitti, usala
        if not has_conflict_second:
            self.final_action = my_second_act
            return True  # Ho cambiato dalla best
        
        # Se anche second_action ha conflitti -> Stay
        self.final_action = 'Stay'
        print(f"[INFO] Drone {self.id}: conflitti su best e second action, uso Stay")
        return True  # Ho cambiato

    def execute_move(self):
        """Aggiorna la propria posizione fisica"""
        if self.final_action is None:
            print(f"[WARNING] Drone {self.id}: final_action is None, staying in place")
            return
        
        d = MOVES_DELTA.get(self.final_action, (0, 0))
        new_pos = (self.pos[0] + d[0], self.pos[1] + d[1])
        
        # Validazione confini
        map_size = self.params['map_size']
        if 0 <= new_pos[0] < map_size and 0 <= new_pos[1] < map_size:
            self.pos = new_pos
        else:
            print(f"[WARNING] Drone {self.id}: movimento fuori confini, rimango in {self.pos}")

    # --- COMUNICAZIONE 2: OSSERVAZIONI ---
    def process_local_observation(self, obs_val):
        """
        1. Aggiorna la propria mappa con il dato sensore locale.
        2. Restituisce il pacchetto dati da inviare al compagno.
        """
        self.belief_map = self.solver_tool.get_updated_belief_map(self.belief_map, self.pos, obs_val)
        
        # Ottieni il piano futuro per l'azione eseguita
        executed_plan = self.future_plans_buffer.get(self.final_action, [])
        
        # Escludiamo la prima azione (già eseguita) dal piano comunicato
        # Il piano originale è [azione_eseguita, t+1, t+2, t+3, t+4]
        # Comunichiamo solo [t+1, t+2, t+3, t+4]
        future_plan = executed_plan[1:] if len(executed_plan) > 1 else []
        
        # Formato pacchetto: (drone_id, posizione, osservazione, piano_futuro)
        return (self.id, self.pos, obs_val, future_plan)

    def receive_remote_observation(self, data_packets):
        """Riceve lista di pacchetti dagli altri droni e aggiorna la mappa e i piani futuri"""
        for data_packet in data_packets:
            # Formato pacchetto: (drone_id, pos, obs, future_plan)
            drone_id, pos, obs, future_plan = data_packet
            
            # Aggiorna belief map con osservazione del partner
            self.belief_map = self.solver_tool.get_updated_belief_map(self.belief_map, pos, obs)
            
            # Salva il piano futuro del partner per il prossimo step di planning
            # Verrà usato per calcolare la penalty repulsiva nel generative_model_G
            self.partner_future_plans[drone_id] = future_plan


# =============================================================================
# 4. FUNZIONI GRAFICHE (FEDELTÀ 100% ORIGINALE)
# =============================================================================

# Disegno griglia, heatmap e percentuali su sfondo
def draw_static_background(surface, p_map, font_cell, params):

    GRID_WIDTH = surface.get_width()
    CELL_SIZE = GRID_WIDTH // params['map_size']
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    surface.fill((255, 255, 255))
    max_prob = p_map.max()
    
    # Ottieni mappa ostacoli se presente
    obstacle_map = initialize_obstacle_map(params) if 'obstacles' in params else None

    for r in range(params['map_size']):
        for c in range(params['map_size']):
            prob = p_map[r, c]

            # Controlla se c'è un ostacolo in questa cella
            if obstacle_map is not None and obstacle_map[r, c] == 1:
                # Cella con ostacolo: colorala interamente di rosso
                color = RED
            else:
                # Heatmap: blu più scuro = probabilità più alta
                color_val = 0
                if max_prob > 1e-9:
                    color_val = int(255 * (prob / max_prob))
                color = (max(0, 255 - color_val), max(0, 255 - color_val), 255)

            # In Pygame: x = colonna (c), y = riga (r)
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)

            # Testo probabilità (non mostrarlo sugli ostacoli)
            if obstacle_map is None or obstacle_map[r, c] == 0:
                text = font_cell.render(f"{prob * 100:.3f}%", True, BLACK)
                surface.blit(text, (c * CELL_SIZE + 5, r * CELL_SIZE + 5))

# Palette di colori per droni multipli
DRONE_COLORS = [
    (255, 0, 0),      # ROSSO
    (0, 0, 200),      # BLU
    (0, 180, 0),      # VERDE
    (255, 140, 0),    # ARANCIONE
    (128, 0, 128),    # VIOLA
    (255, 192, 203),  # ROSA
    (165, 42, 42),    # MARRONE
    (0, 255, 255),    # CIANO
    (255, 255, 0),    # GIALLO
    (128, 128, 128),  # GRIGIO SCURO
]

def get_drone_color(drone_id):
    """Restituisce un colore unico per ogni drone (ciclico se > 10 droni)"""
    return DRONE_COLORS[(drone_id - 1) % len(DRONE_COLORS)]

# Funzioni per disegnare elementi dinamici: droni, target e barra laterale (N droni)
def draw_elements(screen, belief_map, drone_agents, target_pos, params, font_sidebar, font_cell, GRID_WIDTH, CELL_SIZE, stats, SIDEBAR_WIDTH):
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 200, 0)  # Per threshold bar
    GRAY = (200, 200, 200)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)  # Usato per la barra di progresso
    PURPLE = (100, 0, 100)

    # Target (X) - posizione logica (riga, colonna) -> (x, y) Pygame
    tx, ty = target_pos
    target_rect = pygame.Rect(ty * CELL_SIZE, tx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.line(screen, BLACK, target_rect.topleft, target_rect.bottomright, 3)
    pygame.draw.line(screen, BLACK, target_rect.topright, target_rect.bottomleft, 3)

    # Disegna tutti i droni dinamicamente
    for drone in drone_agents:
        dr, dc = drone.pos
        center = (dc * CELL_SIZE + CELL_SIZE // 2, dr * CELL_SIZE + CELL_SIZE // 2)
        color = get_drone_color(drone.id)
        pygame.draw.circle(screen, color, center, CELL_SIZE // 3, 4)
        
        # Disegna l'ID del drone al centro del cerchio
        id_text = font_cell.render(str(drone.id), True, color)
        text_rect = id_text.get_rect(center=center)
        screen.blit(id_text, text_rect)

    # Sidebar - estesa per tutta l'altezza della finestra
    screen_height = screen.get_height()
    sidebar_rect = pygame.Rect(GRID_WIDTH, 0, SIDEBAR_WIDTH, screen_height)
    pygame.draw.rect(screen, GRAY, sidebar_rect)

    # Statistiche
    y_offset = 10
    spacing = 16  # Spacing compatto per supportare più droni

    text_step = font_sidebar.render(f"Step: {stats['step']}", True, BLACK)
    screen.blit(text_step, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing + 5

    # Disegna info per tutti i droni dinamicamente
    for drone in drone_agents:
        drone_id = drone.id
        color = get_drone_color(drone_id)
        
        # Header drone
        text_header = font_sidebar.render(f"=== Drone {drone_id} ===", True, color)
        screen.blit(text_header, (GRID_WIDTH + 10, y_offset))
        y_offset += spacing

        # Simulations
        sims = stats['drones'][drone_id].get('visits', 0)
        text_sims = font_sidebar.render(f"Sims: {sims}", True, BLACK)
        screen.blit(text_sims, (GRID_WIDTH + 20, y_offset))
        y_offset += spacing

        # Tree depth
        depth = stats['drones'][drone_id].get('depth', 0)
        text_depth = font_sidebar.render(f"Depth: {depth}", True, BLACK)
        screen.blit(text_depth, (GRID_WIDTH + 20, y_offset))
        y_offset += spacing

        # Best action
        best_act = stats['drones'][drone_id].get('best', '-')
        best_q = stats['drones'][drone_id].get('best_q', 0)
        text_best = font_sidebar.render(f"Best: {best_act} Q={best_q:.3f}", True, BLACK)
        screen.blit(text_best, (GRID_WIDTH + 20, y_offset))
        y_offset += spacing

        # Second action
        second_act = stats['drones'][drone_id].get('second', '-')
        second_q = stats['drones'][drone_id].get('second_q', 0)
        text_second = font_sidebar.render(f"2nd: {second_act} Q={second_q:.3f}", True, BLACK)
        screen.blit(text_second, (GRID_WIDTH + 20, y_offset))
        y_offset += spacing

        # Final action (executed after conflict resolution)
        final_act = stats['drones'][drone_id].get('final', '-')
        text_final = font_sidebar.render(f"Final: {final_act}", True, BLACK)
        screen.blit(text_final, (GRID_WIDTH + 20, y_offset))
        y_offset += spacing

        # Conflict indicator
        if stats['drones'][drone_id].get('conflict', False):
            text_conflict = font_sidebar.render("⚠ Conflict!", True, (200, 0, 0))
            screen.blit(text_conflict, (GRID_WIDTH + 20, y_offset))
        y_offset += spacing + 2

    # Max probabilità e cella
    max_prob = belief_map.max()
    max_pos = np.unravel_index(np.argmax(belief_map), belief_map.shape)
    text_max = font_sidebar.render(f"Max Prob: {max_prob:.4f}", True, BLACK)
    screen.blit(text_max, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing
    text_max_cell = font_sidebar.render(f"Max Cell: {max_pos}", True, BLACK)
    screen.blit(text_max_cell, (GRID_WIDTH + 20, y_offset))
    y_offset += spacing + 5  # Ridotto da 10 a 5

    # Barra probabilità massima con threshold
    bar_width = SIDEBAR_WIDTH - 40
    pygame.draw.rect(screen, WHITE, (GRID_WIDTH + 20, y_offset, bar_width, 20))
    pygame.draw.rect(screen, BLUE, (GRID_WIDTH + 20, y_offset, bar_width * min(max_prob, 1.0), 20))
    thr_pos = (GRID_WIDTH + 20) + bar_width * 0.95
    pygame.draw.line(screen, GREEN, (thr_pos, y_offset - 3), (thr_pos, y_offset + 23), 2)
    text_thr = font_sidebar.render("Threshold 0.95", True, GREEN)
    screen.blit(text_thr, (GRID_WIDTH + 20, y_offset + 22))
    y_offset += 45  # Ridotto da 50 a 45

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

# Simula sensore reale
def get_real_observation(drone_pos, target_pos, alpha, beta):
    is_target = (drone_pos == target_pos)
    if is_target:
        return 0 if np.random.rand() < beta else 1
    return 1 if np.random.rand() < alpha else 0


# =============================================================================
# 5. MAIN LOOP RISTRUTTURATO PER DECENTRALIZZAZIONE (GRAFICA ORIGINALE)
# =============================================================================

def run_simulation(params):
    pygame.init()

    # Setup schermo con adattamento automatico alla risoluzione disponibile
    map_size = params['map_size']
    num_drones = params['num_drones']
    
    # Ottieni risoluzione schermo disponibile
    display_info = pygame.display.Info()
    available_width = display_info.current_w
    available_height = display_info.current_h
    
    # Riserva spazio per sidebar e margini (ridotti per far stare tutto nello schermo)
    sidebar_w = 480
    margin = 80  # Margine di sicurezza ridotto
    
    # Calcola dimensione finestra massima che sta nello schermo
    max_window_w = available_width - margin
    max_window_h = available_height - margin
    
    # Calcola dimensione massima celle in base allo spazio disponibile
    max_cell_w = (max_window_w - sidebar_w) // map_size
    max_cell_h = max_window_h // map_size
    cell_size = min(max_cell_w, max_cell_h, 80)  # Max 80px per cella
    cell_size = max(cell_size, 25)  # Min 25px per cella
    
    GRID_WIDTH = map_size * cell_size
    screen_w = GRID_WIDTH + sidebar_w
    screen_h = map_size * cell_size
    
    # Assicurati che la finestra non superi lo schermo disponibile
    screen_w = min(screen_w, max_window_w)
    screen_h = min(screen_h, max_window_h)
    
    # Log informazioni visualizzazione
    print(f"\n{'='*30}")
    print(f"Risoluzione schermo: {available_width}x{available_height}")
    print(f"Dimensione mappa: {map_size}x{map_size}")
    print(f"Dimensione celle: {cell_size}px")
    print(f"Finestra creata: {screen_w}x{screen_h}px")
    print(f"={'='*30}\n")

    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Multi-Drone POMCP Decentralized")

    # Font adattati alla dimensione delle celle
    font_cell_size = max(10, min(18, cell_size // 3))
    font_cell = pygame.font.SysFont(None, font_cell_size)
    font_sidebar = pygame.font.SysFont(None, 20)

    # 1. ISTANZIAZIONE ENTITÀ SEPARATE
    # Creiamo N agenti droni dinamicamente
    drones = []
    for i in range(num_drones):
        drone = DroneAgent(i + 1, params['drone_positions'][i], params)
        drones.append(drone)
    target_pos = params['target_pos']
    
    # Setup multiprocessing per N droni
    pool = multiprocessing.Pool(processes=num_drones)
    clock = pygame.time.Clock()
    
    running = True
    auto_mode = False
    step_counter = 0
    auto_timer = 0
    AUTO_INTERVAL = 500  # ms (0.5 secondi)

    background_surface = pygame.Surface((GRID_WIDTH, screen_h))
    force_redraw = True
    is_recording = False
    frames = []
    capture_frame = False  # Flag per catturare frame dopo rendering normale

    # UI Stats (Struttura dati dinamica per N droni)
    ui_stats = {
        'step': 0,
        'drones': {drone.id: {
            'obs': '-',
            'depth': 0,
            'visits': 0,
            'nodes': 0,
            'best': '-',
            'best_q': 0,
            'second': '-',
            'second_q': 0,
            'final': '-',
            'conflict': False
        } for drone in drones}
    }

    try:
        while running:
            if auto_mode:
                current_time = pygame.time.get_ticks()
                if current_time - auto_timer > AUTO_INTERVAL:
                    step_counter += 1
                    print(f"\n--- STEP {step_counter} ---")

                # FASE 1: PIANIFICAZIONE PARALLELA (Agenti autonomi - N droni)
                # Ogni drone prepara il suo pacchetto con info sugli altri droni
                tasks = []
                for drone in drones:
                    # Ottieni lista degli altri droni (escluso se stesso)
                    other_drones = [other for other in drones if other.id != drone.id]
                    task = drone.get_planning_args(other_drones)
                    tasks.append(task)
                
                # Esegui POMCP in parallelo per tutti i droni
                results = pool.map(worker_pomcp_task, tasks)
                
                # I droni ricevono i risultati
                for i, drone in enumerate(drones):
                    drone.set_planning_result(results[i])

                # FASE 2: SCAMBIO INTENZIONI & CONFLITTI
                # Simula scambio messaggi: ogni drone crea il suo pacchetto
                intention_packets = [drone.create_intention_packet() for drone in drones]
                
                # Risoluzione centralizzata dei conflitti
                final_actions = resolve_conflicts_centralized(drones, intention_packets)
                
                # Assegna le azioni finali ai droni
                for drone in drones:
                    drone.final_action = final_actions[drone.id]

                # FASE 3: MOVIMENTO
                for drone in drones:
                    drone.execute_move()

                # FASE 4: SENSING (Simulazione Fisica)
                observations = {}
                for drone in drones:
                    obs = get_real_observation(drone.pos, target_pos, params['real_alpha'], params['real_beta'])
                    observations[drone.id] = obs
                
                # FASE 5: COMUNICAZIONE DATI
                # Ognuno processa il proprio dato e crea un pacchetto
                data_packets = {}
                for drone in drones:
                    data_pkt = drone.process_local_observation(observations[drone.id])
                    data_packets[drone.id] = data_pkt
                
                # Ognuno riceve i pacchetti degli altri
                for drone in drones:
                    other_data = [data_packets[other_id] for other_id in data_packets if other_id != drone.id]
                    drone.receive_remote_observation(other_data)

                # AGGIORNAMENTO DATI PER UI
                # Mappiamo i dati interni degli agenti nel dizionario stats
                ui_stats['step'] = step_counter
                for drone in drones:
                    result = drone.planned_result
                    # Determina se c'è stato un conflitto (final_action != best_action)
                    had_conflict = (drone.final_action != result['best_action'])
                    ui_stats['drones'][drone.id].update({
                        'obs': observations[drone.id],
                        'depth': result['depth'],
                        'visits': result['visits'],
                        'nodes': result['nodes_created'],
                        'best': result['best_action'],
                        'best_q': result['best_q'],
                        'second': result['second_action'],
                        'second_q': result['second_q'],
                        'final': drone.final_action,
                        'conflict': had_conflict
                    })

                capture_frame = is_recording  # Segna che serve catturare questo frame

                # TERMINAZIONE (Threshold Check sulla belief del primo drone)
                if drones[0].belief_map.max() >= 0.95:
                    print("\n TARGET TROVATO! (probabilità > 95%)")
                    auto_mode = False
                
                force_redraw = True
                auto_timer = current_time

            # GESTIONE EVENTI (IDENTICA ALL'ORIGINALE)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: return "quit"
                    if event.key == pygame.K_r: return "restart"
                    if event.key == pygame.K_SPACE: 
                        auto_mode = not auto_mode
                        if auto_mode:
                            print("\n✓ Modalità AUTO POMCP attivata")
                            auto_timer = pygame.time.get_ticks()
                        else:
                            print("\n✓ Modalità AUTO disattivata")
                    if event.key == pygame.K_g:
                        is_recording = not is_recording
                        if is_recording:
                            print("🔴 Registrazione GIF avviata")
                            frames = []
                        else:
                            print("💾 Salvataggio GIF...")
                            filename = f'multi_drone_pomcp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
                            imageio.mimsave(filename, frames, fps=30, loop=0)
                            print(f"✅ GIF salvata: {filename}"); frames = []

            # DISEGNO
            if force_redraw:
                # Usiamo la mappa del primo drone per il background
                draw_static_background(background_surface, drones[0].belief_map, font_cell, params)
                force_redraw = False
                
            screen.fill((255, 255, 255))
            screen.blit(background_surface, (0, 0))
            # Disegnamo gli elementi prendendo le posizioni dagli agenti (N droni)
            draw_elements(
                screen, drones[0].belief_map, drones, target_pos, params,
                font_sidebar, font_cell, GRID_WIDTH, cell_size, ui_stats, sidebar_w
            )
            
            if is_recording:
                pygame.draw.circle(screen, (255, 0, 0), (screen_w - 20, 20), 10)

            pygame.display.flip()

            # Cattura frame dopo rendering (se movimento appena avvenuto)
            if capture_frame:
                rect = pygame.Rect(0, 0, screen_w, screen_h)
                sub = screen.subsurface(rect)
                frame_data = pygame.surfarray.array3d(sub)
                frame_data = np.rot90(frame_data)
                frame_data = np.flipud(frame_data)
                # Duplica frame per mantenere durata reale (0.5s * 30fps = 15 frames)
                for _ in range(15):
                    frames.append(frame_data.copy())
                capture_frame = False

            clock.tick(30)
    finally:
        # Assicurarsi che il pool venga sempre chiuso
        pool.close()
        pool.join()


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
