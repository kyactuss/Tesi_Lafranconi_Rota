#SPIEGAZIONE CODICE:
#Mappa di probabilità con aggiornamento bayesiano, possibilità di introdurre modello del sensore non ideale,
#distribuzione uniforme o gaussiana iniziale, e strategia di movimento ad Asta per più droni.

import pygame
import numpy as np
from scipy.stats import multivariate_normal
import sys
import imageio
from datetime import datetime

# Parametri di default
DEFAULT_MAP_SIZE = 15  # Dimensione lato mappa quadrata
DEFAULT_SENSOR_ALPHA = 0.01  # Imposta qui il falso positivo del sensore
DEFAULT_SENSOR_BETA = 0.01   # Imposta qui il falso negativo del sensore


# --- 1. Funzioni di Configurazione e Inizializzazione ---

def get_user_parameters():
    """
    Interfaccia grafica interattiva per configurare la missione.
    L'utente clicca sulla griglia per selezionare coordinate, preme INVIO per confermare.
    """
    print("=== CONFIGURAZIONE MISSIONE DI RICERCA (GUI) - STRATEGIA AD ASTA ===")
    
    map_size = DEFAULT_MAP_SIZE
    alpha = DEFAULT_SENSOR_ALPHA
    beta = DEFAULT_SENSOR_BETA
    print(f"Parametri: Map Size={map_size}x{map_size}, Alpha={alpha}, Beta={beta}")
    
    # Inizializza pygame
    pygame.init()
    
    # Ottieni dimensioni dello schermo disponibile
    display_info = pygame.display.Info()
    screen_width = display_info.current_w
    screen_height = display_info.current_h
    
    # Calcola dimensioni ottimali per adattarsi allo schermo
    info_height = 150
    available_width = screen_width - 100
    available_height = screen_height - info_height - 150
    
    # Calcola cell_size in base allo spazio disponibile
    max_cell_from_width = available_width // map_size
    max_cell_from_height = available_height // map_size
    cell_size = min(max_cell_from_width, max_cell_from_height, 40)
    cell_size = max(cell_size, 20)
    
    # Calcola margini proporzionalmente
    margin = max(30, min(50, (available_width - map_size * cell_size) // 2))
    
    # Dimensioni finali della finestra
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
    
    # Font adattivi
    font_size_title = max(24, min(32, cell_size + 4))
    font_size_info = max(18, min(24, cell_size - 4))
    font_size_small = max(12, min(18, int(cell_size * 0.5)))
    
    font_title = pygame.font.Font(None, font_size_title)
    font_info = pygame.font.Font(None, font_size_info)
    font_small = pygame.font.Font(None, font_size_small)
    
    def draw_grid(selected_positions, phase_name, phase_color, context_items=None):
        """Disegna la griglia con le posizioni selezionate e gli elementi contestuali"""
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
        """Modalità interattiva per selezionare celle"""
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
    
    # Creazione elenco drone_positions formattato
    drone_positions_list = [[pos[0], pos[1]] for pos in drone_positions]
    
    # Return dizionario compatibile
    return {
        "alpha": alpha,
        "beta": beta,
        "b0": 1,
        "target_pos": target_pos,
        "dist_type": ['uniform', 'gaussian_single', 'gaussian_multi'][map_type-1],
        "dist_params": {'peaks': [{'mean': p['mean'], 'cov': [[p['cov'][0]**2, 0], [0, p['cov'][1]**2]]} for p in map_config['peaks']]},
        "grid_size": (map_size, map_size),
        "threshold_upper": 0.95,
        "initial_drones": drone_positions_list,
        "num_drones": num_drones,
        "obstacles": obstacles
    }


# Funzione per inizializzare la mappa di credenza
def initialize_belief_map(params):
    grid_size = params["grid_size"]
    b0 = params["b0"]
    p_map = None 

    dist_type = params["dist_type"]

    if dist_type == 'uniform':
        cell_prob = b0 / (grid_size[0] * grid_size[1])
        p_map = np.full(grid_size, cell_prob)
    elif dist_type in ('gaussian_single', 'gaussian_multi'):
        x, y = np.mgrid[0:grid_size[0], 0:grid_size[1]]
        coord = np.dstack((x, y))
        p_map = np.zeros(grid_size)

        peaks = params.get("dist_params", {}).get("peaks", [])
        if not peaks:
            cell_prob = b0 / (grid_size[0] * grid_size[1])
            p_map = np.full(grid_size, cell_prob)
        else:
            for peak in peaks:
                mean = peak["mean"]
                cov = peak["cov"]
                rv = multivariate_normal(mean, cov)
                p_map += rv.pdf(coord)

            # Non normalizzare ancora, prima applichiamo gli ostacoli
            if p_map.sum() == 0:
                cell_prob = b0 / (grid_size[0] * grid_size[1])
                p_map = np.full(grid_size, cell_prob)
    else:
        cell_prob = b0 / (grid_size[0] * grid_size[1])
        p_map = np.full(grid_size, cell_prob)
    
    # Gestione ostacoli: azzera probabilità nelle celle con ostacoli
    if 'obstacles' in params and len(params['obstacles']) > 0:
        for obs_pos in params['obstacles']:
            r, c = obs_pos
            p_map[r, c] = 0.0
    
    # Normalizzazione finale
    total_pdf = p_map.sum()
    if total_pdf > 0:
        p_map = (p_map / total_pdf) * b0
    else:
        # Fallback: distribuzione uniforme solo sulle celle libere
        cell_prob = b0 / (grid_size[0] * grid_size[1])
        p_map = np.full(grid_size, cell_prob)
        if 'obstacles' in params and len(params['obstacles']) > 0:
            for obs_pos in params['obstacles']:
                r, c = obs_pos
                p_map[r, c] = 0.0
            total_pdf = p_map.sum()
            if total_pdf > 0:
                p_map = (p_map / total_pdf) * b0
    
    return p_map


# --- 2. Logica Core: Aggiornamento Bayesiano ---

def update_bayesian_map(p_map, inspected_cell, params):
    alpha = params["alpha"]
    beta = params["beta"]
    target_pos = params["target_pos"]
    
    is_target_present = (inspected_cell == target_pos)
    
    # Osservazione del sensore con modello non ideale
    if is_target_present:
        if np.random.rand() < beta: 
            observation_Y = 0  # Falso Negativo
        else: 
            observation_Y = 1  # Osservazione corretta 
    else:
        if np.random.rand() < alpha: 
            observation_Y = 1  # Falso Positivo
        else: 
            observation_Y = 0  # Osservazione corretta
            
    if observation_Y == 1:
        Psi = 1 - beta
        Phi = alpha
    else:
        Psi = beta
        Phi = 1 - alpha

    Omega = Psi - Phi
    p_st = p_map[inspected_cell]
    Z = Phi + Omega * p_st
    
    if Z < 1e-9:  # Evitare divisione per zero
        return p_map

    # Aggiornamento delle probabilità
    p_map_t = (Phi * p_map) / Z
    p_map_t[inspected_cell] = (Psi * p_st) / Z
    
    return p_map_t


def check_decision_thresholds(p_map, params):
    if np.any(p_map >= params["threshold_upper"]):
        cell_idx = np.unravel_index(np.argmax(p_map), p_map.shape)
        cell_number = cell_idx[1] * params["grid_size"][0] + cell_idx[0]
        return f"DECISIONE: PRESENTE nella cella {cell_number} (p_c > {params['threshold_upper']:.2f})"
    return None


# --- 3. Logica strategia di movimento ad Asta ---

def calculate_utility_map(p_map, drone_pos, obstacle_map):
    """Calcola la mappa di utilità U = P/(d+1) per un drone"""
    grid_w, grid_h = p_map.shape
    
    # Calcola distanza di Manhattan per ogni cella
    xx, yy = np.mgrid[0:grid_w, 0:grid_h]
    distance_map = np.abs(xx - drone_pos[0]) + np.abs(yy - drone_pos[1])
    
    # Calcola utilità U = P / (d + 1)
    utility_map = p_map / (distance_map + 1.0)
    
    # Azzera utilità per ostacoli
    if obstacle_map is not None:
        utility_map[obstacle_map == 1] = -np.inf
    
    return utility_map


def create_wish_list(utility_map, drone_pos):
    """Crea lista dei desideri ordinata dalla migliore alla peggiore"""
    grid_w, grid_h = utility_map.shape
    
    # Crea lista di tuple (utilità, posizione)
    wish_list = []
    for r in range(grid_w):
        for c in range(grid_h):
            if utility_map[r, c] > -np.inf:  # Escludi ostacoli
                wish_list.append((utility_map[r, c], (r, c)))
    
    # Ordina per utilità decrescente
    wish_list.sort(reverse=True, key=lambda x: x[0])
    
    return wish_list


def auction_based_planning(p_map, drone_positions, grid_size, obstacle_map=None):
    """
    Sistema di asta per assegnare celle target ai droni.
    Ogni drone crea una lista dei desideri e fa offerte iterative.
    """
    num_drones = len(drone_positions)
    grid_w, grid_h = grid_size
    
    # 1. Calcola utilità e crea liste dei desideri per ogni drone
    drone_data = []
    for idx, pos in enumerate(drone_positions):
        utility_map = calculate_utility_map(p_map, pos, obstacle_map)
        wish_list = create_wish_list(utility_map, pos)
        drone_data.append({
            'id': idx,
            'pos': tuple(pos),
            'wish_list': wish_list,
            'wish_index': 0,  # Indice attuale nella wish list
            'assigned_cell': None,
            'assigned': False
        })
    
    # 2. Dizionario per tracciare le offerte per ogni cella
    # Format: {cell: {'drone_id': id, 'utility': value}}
    cell_bids = {}
    
    # 3. Processo iterativo di bidding
    max_iterations = grid_w * grid_h * num_drones  # Safety limit
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        any_change = False
        
        # Ogni drone fa un'offerta per la sua scelta corrente
        for drone in drone_data:
            if drone['assigned']:
                continue
            
            # Ottieni la cella desiderata corrente
            if drone['wish_index'] >= len(drone['wish_list']):
                # Nessuna cella disponibile, resta fermo
                drone['assigned_cell'] = drone['pos']
                drone['assigned'] = True
                continue
            
            utility, desired_cell = drone['wish_list'][drone['wish_index']]
            
            # Controlla se la cella è la posizione corrente di un altro drone
            is_occupied_by_other = any(
                d['pos'] == desired_cell and d['id'] != drone['id'] 
                for d in drone_data
            )
            
            if is_occupied_by_other:
                # Salta questa cella e passa alla prossima
                drone['wish_index'] += 1
                any_change = True
                continue
            
            # Fa un'offerta per questa cella
            if desired_cell not in cell_bids:
                # Prima offerta per questa cella
                cell_bids[desired_cell] = {
                    'drone_id': drone['id'],
                    'utility': utility
                }
            else:
                # C'è già un'offerta, confronta le utilità
                current_bid = cell_bids[desired_cell]
                
                if utility > current_bid['utility']:
                    # Questo drone vince, scalza il precedente
                    # Il drone scalzato dovrà riprovare
                    for d in drone_data:
                        if d['id'] == current_bid['drone_id']:
                            d['wish_index'] += 1
                            d['assigned'] = False
                            any_change = True
                            break
                    
                    # Aggiorna con la nuova offerta
                    cell_bids[desired_cell] = {
                        'drone_id': drone['id'],
                        'utility': utility
                    }
                else:
                    # Questo drone perde, deve provare la prossima scelta
                    drone['wish_index'] += 1
                    any_change = True
        
        # Assegna temporaneamente le celle ai droni vincenti
        for cell, bid in cell_bids.items():
            drone_id = bid['drone_id']
            drone_data[drone_id]['assigned_cell'] = cell
            drone_data[drone_id]['assigned'] = True
        
        # Se nessun cambiamento, tutti i droni sono assegnati
        if not any_change:
            break
    
    # 4. Previeni swap simultanei
    # Due droni non possono scambiarsi le posizioni
    for i in range(num_drones):
        for j in range(i+1, num_drones):
            drone_i = drone_data[i]
            drone_j = drone_data[j]
            
            if (drone_i['assigned_cell'] == drone_j['pos'] and 
                drone_j['assigned_cell'] == drone_i['pos']):
                # Swap detected! Il drone con indice maggiore resta fermo
                drone_j['assigned_cell'] = drone_j['pos']
    
    # 5. Calcola le mosse effettive (un passo verso la cella assegnata)
    final_moves = []
    for drone in drone_data:
        target_cell = drone['assigned_cell']
        current_pos = drone['pos']
        
        if target_cell == current_pos:
            # Resta fermo
            final_moves.append(list(current_pos))
        else:
            # Muovi di un passo verso il target
            dr = np.sign(target_cell[0] - current_pos[0])
            dc = np.sign(target_cell[1] - current_pos[1])
            
            # Priorità alla direzione con distanza maggiore
            dist_r = abs(target_cell[0] - current_pos[0])
            dist_c = abs(target_cell[1] - current_pos[1])
            
            if dist_r >= dist_c and dr != 0:
                # Muovi verticalmente
                next_pos = (current_pos[0] + dr, current_pos[1])
            elif dc != 0:
                # Muovi orizzontalmente
                next_pos = (current_pos[0], current_pos[1] + dc)
            else:
                # Già nella posizione target
                next_pos = current_pos
            
            # Verifica che la mossa sia valida
            if (0 <= next_pos[0] < grid_w and 0 <= next_pos[1] < grid_h):
                # Controlla ostacoli
                if obstacle_map is not None and obstacle_map[next_pos[0], next_pos[1]] == 1:
                    next_pos = current_pos  # Resta fermo se c'è un ostacolo
                
                final_moves.append(list(next_pos))
            else:
                final_moves.append(list(current_pos))
    
    # 6. Verifica collisioni nella mossa finale
    # Se due droni vanno nella stessa cella, quello con indice minore ha priorità
    for i in range(num_drones):
        for j in range(i+1, num_drones):
            if final_moves[i] == final_moves[j]:
                # Collisione! Il drone j resta fermo
                final_moves[j] = list(drone_positions[j])
    
    return final_moves


# --- 4. Funzioni di Simulazione (Pygame) ---

def draw_static_background(surface, p_map, font_cell, params, obstacle_map):
    """Disegna la griglia, la heatmap e il testo sulla superficie di sfondo"""
    GRID_WIDTH = surface.get_width()  
    CELL_SIZE = GRID_WIDTH // params["grid_size"][0]
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    surface.fill(WHITE)
    max_prob = p_map.max() 
    
    for r in range(params["grid_size"][0]):
        for c in range(params["grid_size"][1]):
            prob = p_map[r, c]
            
            # Check se è un ostacolo
            is_obstacle = obstacle_map[r, c] == 1 if obstacle_map is not None else False
            
            if is_obstacle:
                # Ostacoli: cella nera con "X"
                color = BLACK
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, BLACK, rect, 1)
                
                # Disegna "X" bianca
                text = font_cell.render("X", True, WHITE)
                text_rect = text.get_rect(center=(c * CELL_SIZE + CELL_SIZE//2, r * CELL_SIZE + CELL_SIZE//2))
                surface.blit(text, text_rect)
            else:
                # Heatmap normale
                color_val = 0
                if max_prob > 1e-9: 
                    color_val = int(255 * (prob / max_prob))
                color = (max(0, 255 - color_val), max(0, 255 - color_val), 255)
                
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, BLACK, rect, 1)

                # Renderizza il testo in percentuale
                text = font_cell.render(f"{prob * 100:.3f}%", True, BLACK)
                surface.blit(text, (c * CELL_SIZE + 5, r * CELL_SIZE + 5))


def draw_elements(screen, max_prob, drone_positions, params, font_sidebar, decision, 
                  simulation_started, GRID_WIDTH, CELL_SIZE, auto_mode_active, combined_steps):
    """Disegna gli elementi dinamici (Droni, Target, Sidebar)"""
    SIDEBAR_WIDTH = 400
    grid_w, grid_h = params.get("grid_size", (10, 10))
    
    # Colori
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 200, 0)
    BLUE = (0, 0, 255)
    ORANGE = (255, 165, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    YELLOW = (255, 255, 0)
    GRAY = (200, 200, 200)
    WHITE = (255, 255, 255)

    # Lista colori droni dinamici
    DRONE_COLORS = [RED, GREEN, BLUE, ORANGE, CYAN, MAGENTA, YELLOW, (128, 0, 128), (255, 192, 203), (165, 42, 42)]

    # --- Disegna il Target Reale (come una X) ---
    tx, ty = params["target_pos"]
    target_rect = pygame.Rect(ty * CELL_SIZE, tx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.line(screen, BLACK, target_rect.topleft, target_rect.bottomright, 3)
    pygame.draw.line(screen, BLACK, target_rect.topright, target_rect.bottomleft, 3)

    # --- Disegna i Droni con colori dinamici ---
    for idx, d_pos in enumerate(drone_positions):
        drone_center = (d_pos[1] * CELL_SIZE + CELL_SIZE // 2, d_pos[0] * CELL_SIZE + CELL_SIZE // 2)
        color = DRONE_COLORS[idx % len(DRONE_COLORS)]
        pygame.draw.circle(screen, color, drone_center, CELL_SIZE // 3, 4)

    # --- Disegna la Sidebar ---
    sidebar_rect = pygame.Rect(GRID_WIDTH, 0, SIDEBAR_WIDTH, GRID_WIDTH)
    pygame.draw.rect(screen, GRAY, sidebar_rect)
    
    # Barra p_c max
    text_max_p = font_sidebar.render(f"Max Prob Cella (p_c):", True, BLACK)
    text_max_p_val = font_sidebar.render(f"{max_prob:.6f}", True, BLACK)
    pygame.draw.rect(screen, WHITE, (GRID_WIDTH + 20, 250, SIDEBAR_WIDTH - 40, 30))
    pygame.draw.rect(screen, ORANGE, (GRID_WIDTH + 20, 250, (SIDEBAR_WIDTH - 40) * min(max_prob, 1.0), 30))
    thr_u_pos = (GRID_WIDTH + 20) + (SIDEBAR_WIDTH - 40) * params["threshold_upper"]
    pygame.draw.line(screen, GREEN, (thr_u_pos, 245), (thr_u_pos, 285), 3)

    text_steps = font_sidebar.render(f"Mosse Totali: {combined_steps}", True, BLACK)
    screen.blit(text_steps, (GRID_WIDTH + 20, 410))

    # Testo e colore modalità automatica
    auto_text = "MODALITA' AUTO: "
    if auto_mode_active:
        auto_text += "ATTIVA (Asta)"
        auto_color = GREEN
    else:
        auto_text += "NON ATTIVA"
        auto_color = BLACK
    
    text_auto = font_sidebar.render(auto_text, True, auto_color)
    screen.blit(text_auto, (GRID_WIDTH + 20, 500))
    
    text_start = font_sidebar.render("Premi SPAZIO per Auto-Mode", True, BLACK)
    screen.blit(text_start, (GRID_WIDTH + 20, 540))

    text_gif = font_sidebar.render("Premi 'g' per REC/STOP GIF", True, (100, 0, 100))
    screen.blit(text_gif, (GRID_WIDTH + 20, 580))

    screen.blit(text_max_p, (GRID_WIDTH + 20, 200))
    screen.blit(text_max_p_val, (GRID_WIDTH + 40, 290))
    
    # Info numero droni
    text_drones = font_sidebar.render(f"Droni attivi: {len(drone_positions)}", True, BLACK)
    screen.blit(text_drones, (GRID_WIDTH + 20, 450))


def run_simulation(params):
    pygame.init()
    
    # Setup Schermo
    grid_w = params["grid_size"][0]
    grid_h = params["grid_size"][1]
    
    # Ottiene le dimensioni dello schermo disponibile
    display_info = pygame.display.Info()
    available_width = display_info.current_w
    available_height = display_info.current_h
    
    # Riserva spazio per la sidebar e margini
    sidebar_w = 350
    margin_horizontal = 50
    margin_vertical = 150
    
    # Calcola la dimensione massima delle celle per adattarsi allo schermo
    max_cell_from_width = (available_width - sidebar_w - margin_horizontal) // grid_w
    max_cell_from_height = (available_height - margin_vertical) // grid_h
    cell_size = min(max_cell_from_width, max_cell_from_height, 50)
    cell_size = max(cell_size, 15)
    
    GRID_WIDTH = grid_w * cell_size
    screen_w = GRID_WIDTH + sidebar_w
    screen_h = grid_h * cell_size
    
    min_height = 650
    screen_h = max(screen_h, min_height)
    
    print(f"\n=== Configurazione Display ===")
    print(f"Risoluzione schermo: {available_width}x{available_height}")
    print(f"Dimensione celle: {cell_size}px")
    print(f"Griglia: {GRID_WIDTH}px, Sidebar: {sidebar_w}px")
    print(f"Finestra: {screen_w}x{screen_h}\n")
    
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Simulatore Ricerca con Droni - Strategia ad Asta")
    
    # Font
    font_cell_size = max(10, min(18, cell_size // 3))
    font_cell = pygame.font.SysFont(None, font_cell_size)
    font_sidebar = pygame.font.SysFont(None, 24)
    
    # Setup Stato
    p_map = initialize_belief_map(params)
    
    # Inizializza obstacle_map
    obstacle_map = None
    if 'obstacles' in params and len(params['obstacles']) > 0:
        obstacle_map = np.zeros((grid_w, grid_h), dtype=int)
        for obs_pos in params['obstacles']:
            r, c = obs_pos
            obstacle_map[r, c] = 1
    
    max_prob = p_map.max()
    simulation_started = False

    # Posizioni iniziali dei droni
    drone_positions = [list(pos) for pos in params["initial_drones"]]
    cells_to_inspect = []
    
    background_surface = pygame.Surface((GRID_WIDTH, screen_h))
    force_redraw = True 

    clock = pygame.time.Clock()
    running = True 
    decision = None 
    
    auto_mode_active = False
    auto_move_timer = 0
    AUTO_MOVE_INTERVAL = 1500  # 1.5 secondi

    combined_steps = 0
    is_recording = False
    frames = []

    # Loop di Simulazione
    while running:
        
        if auto_mode_active and not decision:
            current_time = pygame.time.get_ticks()
            
            if current_time - auto_move_timer > AUTO_MOVE_INTERVAL:
                # Pianifica le mosse usando l'algoritmo ad asta
                new_positions = auction_based_planning(
                    p_map,
                    drone_positions,
                    params["grid_size"],
                    obstacle_map=obstacle_map
                )

                for idx, new_pos in enumerate(new_positions):
                    drone_positions[idx] = new_pos
                    cells_to_inspect.append(tuple(new_pos))
                    
                auto_move_timer = current_time
                combined_steps += 1
        
        # Gestione Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit" 
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    is_recording = not is_recording
                    if is_recording:
                        print("🔴 Registrazione avviata")
                        frames = []
                    else:
                        print("💾 Salvataggio GIF in corso")
                        filename = f'simulazione_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
                        imageio.mimsave(filename, frames, fps=30, loop=0)
                        print(f"✅ GIF salvata: {filename}")
                        frames = []

                if event.key == pygame.K_ESCAPE:
                    return "quit" 
                
                if decision:
                    if event.key == pygame.K_r:
                        return "restart" 
                
                elif event.key == pygame.K_SPACE and not decision:
                    if not auto_mode_active:
                        print("Modalità automatica ATTIVATA (Asta).")
                        auto_mode_active = True
                        auto_move_timer = pygame.time.get_ticks() 

                        if not simulation_started:
                            simulation_started = True
                            force_redraw = True 
                    else:
                        print("Modalità automatica DISATTIVATA.")
                        auto_mode_active = False

        # Aggiornamento Stato
        if cells_to_inspect and not decision:
            for cell in cells_to_inspect:
                p_map = update_bayesian_map(p_map, cell, params)
                decision = check_decision_thresholds(p_map, params)
                if decision: 
                    break
            
            max_prob = p_map.max()
            cells_to_inspect = [] 
            force_redraw = True 

        # Disegno Ottimizzato
        if force_redraw:
            draw_static_background(background_surface, p_map, font_cell, params, obstacle_map)
            force_redraw = False 

        screen.blit(background_surface, (0, 0))
        draw_elements(screen, max_prob, drone_positions, params, font_sidebar, decision, 
                     simulation_started, GRID_WIDTH, cell_size, auto_mode_active, combined_steps)
        
        if is_recording:
            pygame.draw.circle(screen, (255, 0, 0), (screen_w - 20, 20), 10)

        pygame.display.flip() 

        if is_recording:
            table_rect = pygame.Rect(0, 0, screen_w, screen_h)
            sub_surface = screen.subsurface(table_rect)
            frame_data = pygame.surfarray.array3d(sub_surface)
            frame_data = np.rot90(frame_data)
            frame_data = np.flipud(frame_data)
            frames.append(frame_data)
            
        clock.tick(30) 


# --- 5. Funzione Main ---

def main():
    while True:
        params = get_user_parameters()
        result = run_simulation(params)
        pygame.quit() 
        if result == "quit":
            print("Simulazione terminata.")
            break
        elif result == "restart":
            print("Riavvio della simulazione...")
            continue

if __name__ == "__main__":
    main()
