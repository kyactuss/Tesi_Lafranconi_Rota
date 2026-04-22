# User Manual: Multi-UAV Target search simulator

This manual provides an overview of how the multi-UAV target search simulator works, simulating five different approaches: Dec-POMCP, Shrinking Dec-POMCP, Centralized POMCP, Auction algorithm and Greedy algorithm.

## PARAMETERS CONFIGURATION

Here the list of parameters configurable in the `DEFAULT_CONFIG` dictionary at the beginning of the code, in the `config_gui.py` file:

* **Map Size:** Defines the dimensions of the spatial grid (square map).
  * *Suggested value:* Between 15 and 30.
* **Alpha and Beta Sensor:** Model the reliability of the onboard sensors. The alpha parameter regulates the probability of false positives, while beta handles false negatives.
  * *Suggested value:* Between 0 (ideal sensor) and 0.1 (noisy sensor).
* **Max Time:** Establishes the maximum computation time (in seconds) allowed for the algorithm to decide the move for each single turn.
  * *Suggested value:* Between 1 and 5.
* **Depth Limit:** Defines the maximum depth reachable by the simulations in the lookahead tree (POMCP).
  * *Suggested value:* Leave the default value. Modify only if you want to reduce the computation depth.
* **Discount Factor:** Determines how much the algorithm values future rewards compared to immediate ones (when value is 1, future rewards are equally evaluated).
  * *Suggested value:* Between 0.9 and 0.99.
* **Exploration Constant:** Balance between exploration and exploitation of the tree.
  * *Suggested value:* Square root of 2 (for default reward values).
* **Reward Alpha:** Encourages the drone to explore new cells with the highest probability, rather than only rewarding the actual discovery of the target.
  * *Note:* Any value entered here is automatically overwritten once the simulator is started.
* **Explorative Reward:** A small bonus assigned to encourage drones to move towards unvisited map cells.
  * *Suggested value:* Between 0.0025 and 0.005.
* **R Target:** The absolute reward value assigned when the target is successfully located in the POMCP.
  * *Suggested value:* 1.
* **Num Drones:** The total number of UAVs.
  * *Suggested value:* Between 2 and 4 (maximum allowed: 8 drones).
* **Obstacle Percentage:** Defines the percentage of the map area that will be occupied by randomly generated obstacles.
  * *Suggested value:* Between 20% and 40%.

> **Excluding Algorithms:** If you want to exclude a specific algorithm from the testing batch, simply open the `simulator.py` file and comment out the desired algorithm within the `ALGORITHMS` list at the beginning of the code.

---

## STARTING SIMULATIONS

To start the simulator, run the `simulator.py` file from your terminal. The system will guide you through the setup by asking for the following inputs:

**1. Number of scenarios to test**
Defines how many different environments to generate and simulate. 
* *Note:* Each scenario is automatically tested across all 5 algorithms.

**2. Configuration Type**
Choose how the scenarios will be built:

* **A - Manual Configuration:** The user interactively sets up the environment by clicking on the grid and using the terminal. You will be asked to define:
  * Drone and target positions.
  * Map type and Gaussian peak locations (standard deviations are entered via terminal).
  * Trace positions and their specific parameters (via terminal):
    * *Von Mises (Vectorial):* Direction in degrees ($0^{\circ}=$ Right, rotating counter-clockwise) and concentration value (suggested between 1 and 5).
    * *Ring (Radial):* Radius (higher value = wider distribution) and variance.
    * *Gaussian (Scalar):* Standard deviations for X and Y.
  * Obstacle positions.

* **B - Automatic Configuration (Recommended):** The system automatically generates randomized scenarios based on the default parameters defined in `config_gui.py`.

**3. Number of parameters to test**
Defines how many different values of a specific tuning parameter (at the moment `reward_alpha`) you want to evaluate.
* *Note:* The total number of simulations run will be calculated as: `(N. of parameters to test) x (N. of scenarios) x (N. of algorithms)`.

---

## SIMULATION RESULTS

* **Execution (No GUI):** When the Automatic Configuration is selected, the graphical interface is intentionally disabled to maximize computation speed. You will monitor the progress directly via the terminal, which will output the current scenario number being tested and an estimated average time to completion.
  * **IMPORTANT NOTE:** Ensure that your computer's automatic sleep or standby mode is disabled during the simulations.
* **Excel Reports:** Once simulations are finished, the system automatically generates a dedicated folder. Inside, you will find Excel spreadsheets containing all the tested parameters, metrics and performance comparisons across the 5 different algorithms.
* **Early Termination:** If you need to stop the testing process before it finishes, simply press `CTRL+C` in the terminal. The simulator will save all the data and results processed up to that exact moment into the Excel file.
* **Scenario Reconstruction:** Because automatic scenarios are generated without a GUI, you might want to visually analyze a specific test case afterward. By launching the `RebuildMap.py` script, you can select the generated Excel file, input the specific Scenario ID you wish to review and the system will graphically recreate the exact map.