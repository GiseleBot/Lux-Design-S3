from lux.utils import direction_to
import sys
import numpy as np
import logging
from baselogic import save_relic_nodes, get_unit_data, find_nearest_relic_node, explore, attack_nearest_enemy, move_toward_target
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        self.unit_explore_locations = dict()

        # Explore
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()

        # Attack
        self.enemy_positions = []
        self.discovered_enemy_ids = set()

    


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        return self.custom_logic(step, obs, remainingOverageTime)
    
    def default(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(
            obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(
            obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        # shape (max_units, 1)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        observed_relic_node_positions = np.array(
            obs["relic_nodes"])  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(
            obs["relic_nodes_mask"])  # shape (max_relic_nodes, )
        # points of each team, team_points[self.team_id] is the points of the your team
        team_points = np.array(obs["team_points"])

        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match

        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(
                    observed_relic_node_positions[id])

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(
                    unit_pos[1] - nearest_relic_node_position[1])

                # if close to the relic node we want to hover around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(
                        unit_pos, nearest_relic_node_position), 0, 0]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(
                        0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(
                    unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        return actions
    
    def phteven(self, step: int, obs, remainingOverageTime: int = 60):
        # shape (max_units, )
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(
            obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

        # shape (max_units, )
        enemy_mask = np.array(obs["units_mask"][self.opp_team_id])
        enemy_positions = np.array(
            obs["units"]["position"][self.opp_team_id])  # shape (max_units, 2)

        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        visible_enemy_units = set(np.where(enemy_mask)[0])  # visible enemy units

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # Check for enemy kills
        if len(visible_enemy_units) == 0:
            # No enemies visible, move randomly
            for unit_id in available_unit_ids:
                unit_pos = unit_positions[unit_id]
                if step % 17 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(
                        0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(
                    unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
            return actions

        # If there are enemies, attack the nearest one
        self.discovered_enemy_ids = set()
        self.enemy_positions = []
        for id in visible_enemy_units:
            self.discovered_enemy_ids.add(id)
            self.enemy_positions.append(enemy_positions[id])

        # Move towards the nearest enemy or attack
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            if len(self.enemy_positions) > 0:
                nearest_enemy_pos = self.enemy_positions[0]
                manhattan_distance = abs(
                    unit_pos[0] - nearest_enemy_pos[0]) + abs(unit_pos[1] - nearest_enemy_pos[1])

                if manhattan_distance <= 4:
                    # Attack enemy
                    actions[unit_id] = [5, nearest_enemy_pos[0] -
                                        unit_pos[0], nearest_enemy_pos[1] - unit_pos[1]]
                else:
                    # Move towards enemy
                    actions[unit_id] = [direction_to(
                        unit_pos, nearest_enemy_pos), 0, 0]

        return actions
    
    def balance(self, step: int, obs, remainingOverageTime: int = 60):
        # Energy
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        
        # Units
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(
            obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        
        # Enemies
        enemy_mask = np.array(obs["units_mask"][self.opp_team_id])
        enemy_positions = np.array(
            obs["units"]["position"][self.opp_team_id])  # shape (max_units, 2)
        visible_enemy_units = set(
            np.where(enemy_mask)[0])  # visible enemy units
        
        # Relics
        observed_relic_node_positions = np.array(
            obs["relic_nodes"])  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(
            obs["relic_nodes_mask"])  # shape (max_relic_nodes, )
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

        # Points of each team, team_points[self.team_id] is the points of the your team
        team_points = np.array(obs["team_points"])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # if there are no visible enemy units, we want to explore the map
        if len(visible_enemy_units) > 0:
            for id in visible_enemy_units:
                if id not in self.discovered_enemy_ids:
                    self.discovered_enemy_ids.add(id)
                    self.enemy_positions.append(enemy_positions[id])
            for unit_id in available_unit_ids:
                unit_pos = unit_positions[unit_id]
                if len(self.enemy_positions) > 0:
                    nearest_enemy_node_position = self.enemy_positions[0]
                    # logger.debug( f"nearest_enemy_node_position: {nearest_enemy_node_position}")
                    manhattan_distance = abs(unit_pos[0] - nearest_enemy_node_position[0]) + abs(
                        unit_pos[1] - nearest_enemy_node_position[1])
                    # if close to the relic node we want to hover around it and hope to gain points
                    if manhattan_distance <= 4:
                        actions[unit_id] = [5, nearest_enemy_node_position[0] - unit_pos[0], nearest_enemy_node_position[1] - unit_pos[1]]

        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(
                    observed_relic_node_positions[id])

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(
                    unit_pos[1] - nearest_relic_node_position[1])

                # if close to the relic node we want to hover around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(
                        unit_pos, nearest_relic_node_position), 0, 0]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(
                        0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(
                    unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        return actions
    
    def attack(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        enemy_mask = np.array(
            obs["units_mask"][self.opp_team_id])  # shape (max_units, )
        unit_mask = np.array(
            obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(
            obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        enemy_positions = np.array(
            obs["units"]["position"][self.opp_team_id])  # shape (max_units, 2)
        # shape (max_units, 1)
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible enemy units
        visible_enemy_units = set(np.where(enemy_mask)[0])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # if there are no visible enemy units, we want to explore the map
        if len(visible_enemy_units) > 0:

            for id in visible_enemy_units:
                if id not in self.discovered_enemy_ids:
                    self.discovered_enemy_ids.add(id)
                    self.enemy_positions.append(enemy_positions[id])

            for unit_id in available_unit_ids:
                unit_pos = unit_positions[unit_id]
                if len(self.enemy_positions) > 0:

                    nearest_enemy_node_position = self.enemy_positions[0]

                    # logger.debug( f"nearest_enemy_node_position: {nearest_enemy_node_position}")

                    manhattan_distance = abs(unit_pos[0] - nearest_enemy_node_position[0]) + abs(
                        unit_pos[1] - nearest_enemy_node_position[1])

                    # if close to the relic node we want to hover around it and hope to gain points
                    if manhattan_distance <= 4:
                        actions[unit_id] = [5, nearest_enemy_node_position[0] -
                                            unit_pos[0], nearest_enemy_node_position[1] - unit_pos[1]]
                    else:
                        # otherwise we want to move towards the relic node
                        actions[unit_id] = [direction_to(
                            unit_pos, nearest_enemy_node_position), 0, 0]
        else:
            for unit_id in available_unit_ids:
                unit_pos = unit_positions[unit_id]
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if step % 6 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(
                        0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(
                    unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        
        return actions
    
    def custom_logic(self, step, obs, remainingOverageTime=60):
        # Get unit data
        unit_data = get_unit_data(self.team_id, self.opp_team_id, obs)
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # Assign unit data
        unit_mask = unit_data["unit_mask"]
        unit_positions = unit_data["unit_positions"]
        unit_energys = unit_data["unit_energys"]
        enemy_mask = unit_data["enemy_mask"]
        enemy_positions = unit_data["enemy_positions"]
        available_unit_ids = unit_data["available_unit_ids"]
        visible_enemy_units = unit_data["visible_enemy_units"]
        observed_relic_node_positions = unit_data["observed_relic_node_positions"]
        observed_relic_nodes_mask = unit_data["observed_relic_nodes_mask"]
        visible_relic_node_ids = unit_data["visible_relic_node_ids"]
        team_points = unit_data["team_points"]

        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]

            # Save relic nodes
            saved_relic_data = save_relic_nodes(unit_id, visible_relic_node_ids, observed_relic_node_positions, self.discovered_relic_nodes_ids, self.relic_node_positions)
            self.relic_node_positions = saved_relic_data["relic_node_positions"]
            self.discovered_relic_nodes_ids = saved_relic_data["discovered_relic_nodes_ids"]

            # Get nearest relic node and actions data
            actions_data = find_nearest_relic_node(unit_pos, self.relic_node_positions)
            movement_type = actions_data["movement_type"]
            destination = actions_data["actions_data"]

            if movement_type == "move_toward":
                # Move towards the relic node
                actions[unit_id] = [direction_to(unit_pos, destination), 0, 0]
            elif movement_type == "close_to_relic":
                # If close to the relic node, move randomly
                actions[unit_id] = [destination, 0, 0]
            else:
                # no relic nodes found, explore map
                if step % 16 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]

        return actions
