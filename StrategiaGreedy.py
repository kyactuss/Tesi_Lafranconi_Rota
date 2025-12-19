#SPIEGAZIONE CODICE:
#Mappa di probabilità con aggiornamento bayesiano, possibilità di introdurre modello del sensore non ideale,
#distribuzione uniforme o gaussiana iniziale, e strategia di movimento Greedy per più droni.

import pygame
import numpy as np
from scipy.stats import multivariate_normal
import sys
import imageio
from datetime import datetime

MAP_SIZE = 10  # Dimensione lato mappa quadrata
SENSOR_ALPHA = 0.00  # Imposta qui il falso positivo del sensore
SENSOR_BETA = 0.00   # Imposta qui il falso negativo del sensore
LOCAL_SUM_THRESHOLD = 0.001  #minimo della somma locale delle probabilità, se non rispettata si segue il massimo globale


# --- 1. Funzioni di Configurazione e Inizializzazione ---

def get_user_parameters():
    print("--- Configurazione Nuova Simulazione ---")
    alpha = None
    beta = None
    b0 = None

    # Dimensione griglia per verifica input utente
    grid_size = (MAP_SIZE, MAP_SIZE)
    grid_w, grid_h = grid_size

    # Funzione che garantisce input intero valido entro range [lo, hi]
    def ask_int_in_range(prompt: str, lo: int, hi: int) -> int:
        while True:
            try:                
                v = int(input(f"{prompt} [{lo}-{hi}]: "))
                if lo <= v <= hi:
                    return v
                print(f"Valore fuori range. Inserisci un intero tra {lo} e {hi}.")
            except ValueError:      #Gestione errore input non intero
                print("Valore non valido. Inserisci un intero.")

    def ask_sigma_pair(prompt: str):
        while True:
            raw_value = input(f"{prompt} (formato sigma_x,sigma_y): ").strip()
            try:
                sx_str, sy_str = raw_value.split(',')
                sx = float(sx_str.strip())
                sy = float(sy_str.strip())
                if sx > 0 and sy > 0:
                    return sx, sy
                print("Entrambe le deviazioni standard devono essere > 0.")
            except ValueError:
                print("Formato non valido. Usa due numeri come '2.0,1.5'.")

    def ask_positive_int(prompt: str) -> int:
        while True:
            try:
                v = int(input(f"{prompt} (>=1): "))
                if v >= 1:
                    return v
                print("Inserisci un intero maggiore o uguale a 1.")
            except ValueError:
                print("Valore non valido. Inserisci un intero.")

    def ask_coord_pair(prompt: str):
        while True:
            raw_value = input(f"{prompt} (formato x,y): ").strip()
            try:
                x_str, y_str = raw_value.split(',')
                x = int(x_str.strip())
                y = int(y_str.strip())
                if 0 <= x < grid_w and 0 <= y < grid_h:
                    return (x, y)
                print(f"Coordinate fuori range. Usa valori tra 0 e {grid_w - 1} per X e 0 e {grid_h - 1} per Y.")
            except ValueError:
                print("Formato non valido. Inserisci due interi come 'x,y'.")
                
    #Impostazioni Sensore
    alpha = SENSOR_ALPHA
    beta = SENSOR_BETA
    b0 = 1

    #Inserimento Posizione Droni Iniziali
    print(f"\nPosiziona il DRONE 1 iniziale (coordinate tra 0 e {grid_w - 1}):")
    drone_start_1 = ask_coord_pair("  Drone 1")

    print(f"\nPosiziona il DRONE 2 iniziale (coordinate tra 0 e {grid_w - 1} e diversi dal Drone 1):")
    while True:
        drone_start_2 = ask_coord_pair("  Drone 2")
        if drone_start_2 != drone_start_1:
            break
        print("Le coordinate dei due droni devono essere diverse. Riprova.")

    #Inserimento Posizione Target
    print("\nPosiziona il VERO target:")
    target_pos = ask_coord_pair("  Target")
    
    #Scelta Distribuzione Probabilità Iniziale
    print("\nScegli la distribuzione di probabilità iniziale:")
    print(" 1) Uniforme")
    print(" 2) Gaussiana singola")
    print(" 3) Gaussiana multipla")
    dist_choice = ask_int_in_range("Seleziona distribuzione", 1, 3)

    dist_type_map = {
        1: 'uniform',
        2: 'gaussian_single',
        3: 'gaussian_multi'
    }
    dist_type = dist_type_map[dist_choice]
    dist_params = {}

    if dist_type in ('gaussian_single', 'gaussian_multi'):
        dist_params['peaks'] = []
        num_peaks = 1 if dist_type == 'gaussian_single' else ask_positive_int("Quante gaussiane vuoi inserire")

        for idx in range(num_peaks):
            print(f"\nConfigura la gaussiana #{idx + 1}:")
            mean = ask_coord_pair("  Centro (x,y)")
            sigma_x, sigma_y = ask_sigma_pair("  Deviazioni standard")
            cov_matrix = [[sigma_x ** 2, 0], [0, sigma_y ** 2]]
            dist_params['peaks'].append({'mean': mean, 'cov': cov_matrix})
    
    #Return, restituisce i valori della funzione nel main
    return {
        "alpha": alpha,
        "beta": beta,
        "b0": b0,
        "target_pos": target_pos,
        "dist_type": dist_type,
        "dist_params": dist_params,
        "grid_size": grid_size,
        "threshold_upper": 0.95,          #Valore soglia superiore per decisione presenza target
        "initial_drones": [[drone_start_1[0], drone_start_1[1]], [drone_start_2[0], drone_start_2[1]]]
    }

#Funzione per inizializzare la mappa di credenza
def initialize_belief_map(params):
    grid_size = params["grid_size"]
    b0 = params["b0"]
    p_map = None 

    dist_type = params["dist_type"]

    if dist_type == 'uniform':
        cell_prob = b0 / (grid_size[0] * grid_size[1])
        p_map = np.full(grid_size, cell_prob)               #Funzione per creare array di dimensione grid_size con valori uguali a cell_prob
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

            total_pdf = p_map.sum()
            if total_pdf > 0:
                p_map = (p_map / total_pdf) * b0
            else:
                cell_prob = b0 / (grid_size[0] * grid_size[1])
                p_map = np.full(grid_size, cell_prob)
    else:
        cell_prob = b0 / (grid_size[0] * grid_size[1])
        p_map = np.full(grid_size, cell_prob)
    
    return p_map







# --- 2. Logica Core: Aggiornamento Bayesiano ---

def update_bayesian_map(p_map, inspected_cell, params):
    alpha = params["alpha"]
    beta = params["beta"]
    target_pos = params["target_pos"]
    
    is_target_present = (inspected_cell == target_pos)    #Assegno True o False in base alla condizione

    #Osservazione del sensore con modello non ideale
    if is_target_present:
        if np.random.rand() < beta: observation_Y = 0       #Falso Negativo
        else: observation_Y = 1                             #Osservazione corretta 
    else:
        if np.random.rand() < alpha: observation_Y = 1      #Falso Positivo
        else: observation_Y = 0                             #Osservazione corretta
            
    if observation_Y == 1:
        Psi = 1 - beta
        Phi = alpha
    else:
        Psi = beta
        Phi = 1 - alpha

    #...passaggi intermedi per arrivare a formula finale...
    Omega = Psi - Phi
    p_st = p_map[inspected_cell]
    Z = Phi + Omega * p_st
    
    if Z < 1e-9:        #Evitare divisione per zero
        return p_map

    #Aggiornamento delle probabilità, formula finale (n.3 del paper "Analysis of search decision")
    p_map_t = (Phi * p_map) / Z                     #Aggiornamento celle non ispezionate
    p_map_t[inspected_cell] = (Psi * p_st) / Z      #Aggiornamento cella ispezionata
    
    return p_map_t

# Funzione per controllare soglie di decisione
def check_decision_thresholds(p_map, params):
    if np.any(p_map >= params["threshold_upper"]):
        cell_idx = np.unravel_index(np.argmax(p_map), p_map.shape)
        cell_number = cell_idx[1] * params["grid_size"][0] + cell_idx[0]
        return f"DECISIONE: PRESENTE nella cella {cell_number} (p_c > {params['threshold_upper']:.2f})"
    return None






#---3. Logica strategia di movimento Greedy coerente ---

def get_next_greedy_move(p_map, drone_pos, grid_size, local_sum_threshold=LOCAL_SUM_THRESHOLD, blocked_cells=None):
    """Valuta N/S/O/E/Stay in base alle probabilità e, se necessario, segue il massimo globale."""

    if blocked_cells is None:
        blocked_cells = set()
    else:
        blocked_cells = set(blocked_cells)

    def _step_towards(start, target):
        if start == target:
            return list(start)
        dr = target[0] - start[0]
        dc = target[1] - start[1]
        move = [start[0], start[1]]
        if abs(dr) >= abs(dc) and dr != 0:
            move[0] += int(np.sign(dr))
        elif dc != 0:
            move[1] += int(np.sign(dc))
        return move

    def _global_best_cell():
        return tuple(np.unravel_index(np.argmax(p_map), p_map.shape))

    grid_w, grid_h = grid_size
    neighbors = [
        (drone_pos[0] - 1, drone_pos[1]),  # Nord
        (drone_pos[0] + 1, drone_pos[1]),  # Sud
        (drone_pos[0], drone_pos[1] - 1),  # Ovest
        (drone_pos[0], drone_pos[1] + 1),  # Est
        (drone_pos[0], drone_pos[1]),      # Stay
    ]

    candidates = []
    for nr, nc in neighbors:
        if not (0 <= nr < grid_w and 0 <= nc < grid_h):
            continue
        if (nr, nc) in blocked_cells:
            continue
        candidates.append(((nr, nc), p_map[nr, nc]))

    if candidates:
        local_sum = sum(prob for _, prob in candidates)
        if local_sum >= local_sum_threshold:
            best_cell, _ = max(candidates, key=lambda item: item[1])
            return list(best_cell)

    fallback_cell = _global_best_cell()         
    step = _step_towards(drone_pos, fallback_cell)
    if tuple(step) in blocked_cells:
        return list(drone_pos)
    return step


def plan_multi_drone_moves(p_map, drone_positions, grid_size, local_sum_threshold=LOCAL_SUM_THRESHOLD):
    """Restituisce una lista di nuove posizioni evitando collisioni e swap."""
    new_positions = []
    current_positions = [tuple(pos) for pos in drone_positions]

    for pos in drone_positions:
        blocked = set(current_positions)
        blocked.discard(tuple(pos))  # un drone può restare fermo nella sua cella
        blocked.update(tuple(p) for p in new_positions)

        candidate = get_next_greedy_move(
            p_map,
            pos,
            grid_size,
            local_sum_threshold=local_sum_threshold,
            blocked_cells=blocked
        )

        if tuple(candidate) in blocked:
            candidate = list(pos)

        new_positions.append(candidate)

    return new_positions







# --- 4. Funzioni di Simulazione (Pygame) ---

def draw_static_background(surface, p_map, font_cell, params):
    """
    Funzione che disegna la griglia, la heatmap e il testo
    solo una volta sulla superficie di sfondo.
    """
    GRID_WIDTH = surface.get_width()  
    CELL_SIZE = GRID_WIDTH // params["grid_size"][0]
    BLACK = (0, 0, 0)

    surface.fill((255, 255, 255)) # Pulisce lo sfondo
    max_prob = p_map.max() 
    
    for r in range(params["grid_size"][0]):
        for c in range(params["grid_size"][1]):
            prob = p_map[r, c]
            
            # Heatmap
            color_val = 0
            if max_prob > 1e-9: 
                color_val = int(255 * (prob / max_prob))
            color = (max(0, 255 - color_val), max(0, 255 - color_val), 255)
            
            # In Pygame: x = colonna (c), y = riga (r)
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, BLACK, rect, 1) # Bordo

            # Renderizza il testo in percentuale
            text = font_cell.render(f"{prob * 100:.3f}%", True, BLACK)
            surface.blit(text, (c * CELL_SIZE + 5, r * CELL_SIZE + 5))

def draw_elements(screen, max_prob, drone_positions, params, font_sidebar, decision, simulation_started, GRID_WIDTH, CELL_SIZE, auto_mode_active, combined_steps):
    """
    Funzione che disegna solo gli elementi "dinamici"
    (Droni, Target, Sidebar) che cambiano ad ogni frame.
    """
    SIDEBAR_WIDTH = 400
    grid_w, grid_h = params.get("grid_size", (10, 10))
    
    # Colori
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 200, 0)
    BLUE = (0, 0, 255)
    GRAY = (200, 200, 200)
    WHITE = (255, 255, 255)

    # --- Disegna il Target Reale (come una X) ---
    tx, ty = params["target_pos"]
    target_rect = pygame.Rect(ty * CELL_SIZE, tx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.line(screen, BLACK, target_rect.topleft, target_rect.bottomright, 3)
    pygame.draw.line(screen, BLACK, target_rect.topright, target_rect.bottomleft, 3)

    # --- Disegna i Droni ---
    drone_colors = [RED, BLUE]
    for idx, d_pos in enumerate(drone_positions):
        drone_center = (d_pos[1] * CELL_SIZE + CELL_SIZE // 2, d_pos[0] * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, drone_colors[idx % len(drone_colors)], drone_center, CELL_SIZE // 3, 4)

    # --- Disegna la Sidebar ---
    sidebar_rect = pygame.Rect(GRID_WIDTH, 0, SIDEBAR_WIDTH, GRID_WIDTH)
    pygame.draw.rect(screen, GRAY, sidebar_rect)
    
    # Barra p_c max
    text_max_p = font_sidebar.render(f"Max Prob Cella (p_c):", True, BLACK)
    text_max_p_val = font_sidebar.render(f"{max_prob:.6f}", True, BLACK)
    pygame.draw.rect(screen, WHITE, (GRID_WIDTH + 20, 250, SIDEBAR_WIDTH - 40, 30))
    pygame.draw.rect(screen, (255,165,0), (GRID_WIDTH + 20, 250, (SIDEBAR_WIDTH - 40) * min(max_prob, 1.0), 30))
    thr_u_pos = (GRID_WIDTH + 20) + (SIDEBAR_WIDTH - 40) * params["threshold_upper"]
    pygame.draw.line(screen, GREEN, (thr_u_pos, 245), (thr_u_pos, 285), 3)

    text_steps = font_sidebar.render(f"Mosse Totali: {combined_steps}", True, BLACK)
    screen.blit(text_steps, (GRID_WIDTH + 20, 410))

    # Testo e colore modalità automatica
    auto_text = "MODALITA' AUTO: "
    if auto_mode_active:
        auto_text += "ATTIVA (Greedy)"
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
    
    # Disegna il messaggio di esito o riavvio se presente

def run_simulation(params):
    
    pygame.init()
    
    # Setup Schermo
    grid_w = params["grid_size"][0]
    grid_h = params["grid_size"][1]
    cell_size = 60
    sidebar_w = 400
    GRID_WIDTH = grid_w * cell_size
    screen_w = GRID_WIDTH + sidebar_w
    screen_h = grid_h * cell_size
    
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Simulatore Ricerca con Droni - Strategia Greedy")
    
    font_cell = pygame.font.SysFont(None, 18) 
    font_sidebar = pygame.font.SysFont(None, 24)  # Font più piccolo per la sidebar
    
    # Setup Stato
    p_map = initialize_belief_map(params)
    
    max_prob = p_map.max()  #Probabilità massima iniziale
    
    simulation_started = False  # Flag per avvio simulazione

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
    AUTO_MOVE_INTERVAL = 1500 # 1.5 secondi in millisecondi, tra una mossa e l'altra

    combined_steps = 0

    is_recording = False
    frames = []


    # Loop di Simulazione
    while running:
        
        if auto_mode_active and not decision:           # Movimento Automatico
            current_time = pygame.time.get_ticks()      # Tempo attuale in ms
            
            if current_time - auto_move_timer > AUTO_MOVE_INTERVAL:  #orologio - timestamp dell'ultima mossa > tempo che decidiamo noi
                
                new_positions = plan_multi_drone_moves(
                    p_map,
                    drone_positions,
                    params["grid_size"]
                )

                for idx, new_pos in enumerate(new_positions):
                    drone_positions[idx] = new_pos
                    cells_to_inspect.append(tuple(new_pos))
                    
                # Resetta il timer per il prossimo intervallo
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
                        frames = [] # Reset buffer
                    else:
                        print("💾 Salvataggio GIF in corso")
                        # Genera nome file
                        filename = f'simulazione_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
                        # Salva
                        imageio.mimsave(filename, frames, fps=30, loop=0)
                        print(f"✅ GIF salvata: {filename}")
                        frames = [] # Libera memoria

                if event.key == pygame.K_ESCAPE:
                    return "quit" 
                
                if decision:
                    if event.key == pygame.K_r:
                        return "restart" 
                
                elif event.key == pygame.K_SPACE and not decision:
                    if not auto_mode_active:
                        print("Modalità automatica ATTIVATA (Greedy).")
                        auto_mode_active = True
                        auto_move_timer = pygame.time.get_ticks() 

                        if not simulation_started:
                            simulation_started = True
                            force_redraw = True 
                    else:
                        print("Modalità automatica DISATTIVATA.")
                        auto_mode_active = False

                # Nessun controllo manuale: tutte le mosse derivano dal planner auto

        # --- Aggiornamento Stato ---
        if cells_to_inspect and not decision:
            
            for cell in cells_to_inspect:
                p_map = update_bayesian_map(p_map, cell, params)
                decision = check_decision_thresholds(p_map, params)
                if decision: break
            
            max_prob = p_map.max()
            
            cells_to_inspect = [] 
            force_redraw = True 

        # --- Disegno Ottimizzato ---
        
        if force_redraw:
            draw_static_background(background_surface, p_map, font_cell, params)
            force_redraw = False 

        screen.blit(background_surface, (0, 0))
        
        draw_elements(screen, max_prob, drone_positions, params, font_sidebar, decision, simulation_started, GRID_WIDTH, cell_size, auto_mode_active, combined_steps)
        
        if is_recording:
            # Disegna un pallino rosso in alto a destra come indicatore di REC
            pygame.draw.circle(screen, (255, 0, 0), (screen_w - 20, 20), 10)


        pygame.display.flip() 

        if is_recording:
            # 1. Cattura l'intera schermata (griglia + barra laterale)
            table_rect = pygame.Rect(0, 0, screen_w, screen_h)
            sub_surface = screen.subsurface(table_rect)
            
            # 2. Estrae i pixel
            frame_data = pygame.surfarray.array3d(sub_surface)
            
            # 3. Corregge rotazione e flip (Pygame -> ImageIO format)
            frame_data = np.rot90(frame_data)
            frame_data = np.flipud(frame_data)
            
            # 4. Aggiunge al buffer
            frames.append(frame_data)
            
        clock.tick(30) 


# --- 5. Funzione Main (Gestisce il loop "Riavvia") ---

def main():
    while True:
        params = get_user_parameters()    #assegno al dizionario params, tutti i valori che inserisco da utente
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
