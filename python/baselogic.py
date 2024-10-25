from lux.utils import direction_to
import sys
import numpy as np
import logging

# GET UNIT DATA
def get_unit_data(team_id, opp_team_id, obs):
    unit_mask =                         np.array(obs["units_mask"][team_id])
    unit_positions =                    np.array(obs["units"]["position"][team_id])  # shape (max_units, 2)
    unit_energys =                      np.array(obs["units"]["energy"][team_id])

    enemy_mask =                        np.array(obs["units_mask"][opp_team_id])
    enemy_positions =                   np.array(obs["units"]["position"][opp_team_id])  # shape (max_units, 2)

    available_unit_ids =                np.where(unit_mask)[0]
    visible_enemy_units =               set(np.where(enemy_mask)[0])

    observed_relic_node_positions =     np.array(obs["relic_nodes"])
    observed_relic_nodes_mask =         np.array(obs["relic_nodes_mask"])
    visible_relic_node_ids =            set(np.where(observed_relic_nodes_mask)[0])
    team_points =                       np.array(obs["team_points"])


    return {
        "unit_mask": unit_mask,
        "unit_positions": unit_positions,
        "unit_energys": unit_energys,
        "enemy_mask": enemy_mask,
        "enemy_positions": enemy_positions,
        "available_unit_ids": available_unit_ids,
        "visible_enemy_units": visible_enemy_units,
        "observed_relic_node_positions": observed_relic_node_positions,
        "observed_relic_nodes_mask": observed_relic_nodes_mask,
        "visible_relic_node_ids": visible_relic_node_ids,
        "team_points": team_points
    }

# FIND RELIC NODES
def find_nearest_relic_node(unit_pos, relic_node_positions):
        movement_type = "random"
        actions_data = None
        if len(relic_node_positions) > 0:

            nearest_node = min(relic_node_positions, key=lambda node: abs(unit_pos[0] - node[0]) + abs(unit_pos[1] - node[1]))
            manhattan_distance = abs(unit_pos[0] - nearest_node[0]) + abs(unit_pos[1] - nearest_node[1])

            # Default action is no movement (0)
            isRandom = False
            if manhattan_distance <= 4:
                # If close to the relic node, move randomly
                random_direction = np.random.randint(0, 5)
                actions_data = random_direction
                movement_type = "close_to_relic"
            else:
                # Move towards the relic node
                actions_data = nearest_node
                movement_type = "move_toward"

            return {
                'actions_data': actions_data,
                'movement_type': movement_type
            }

# SAVE RELIC NODES
def save_relic_nodes(id,visible_relic_node_ids, observed_relic_node_positions, discovered_relic_nodes_ids, relic_node_positions):
    for id in visible_relic_node_ids:
        if id not in discovered_relic_nodes_ids:
            discovered_relic_nodes_ids.add(id)
            relic_node_positions.append(observed_relic_node_positions[id])
    return {
        'discovered_relic_nodes_ids': discovered_relic_nodes_ids,
        'relic_node_positions': relic_node_positions
    }

# EXPLORE MAP
def explore(unit_id, unit_pos, step, unit_explore_locations, map_width, map_height):
    if step % 20 == 0 or unit_id not in unit_explore_locations:
        rand_loc = (np.random.randint(0, map_width), np.random.randint(0, map_height))
        unit_explore_locations[unit_id] = rand_loc

    return direction_to(unit_pos, unit_explore_locations[unit_id])

# ATTACK NEAREST ENEMY
def attack_nearest_enemy(unit_pos, enemy_positions):
    if len(enemy_positions) == 0:
        return None, float('inf')

    nearest_enemy = min(enemy_positions, key=lambda enemy: abs(unit_pos[0] - enemy[0]) + abs(unit_pos[1] - enemy[1]))
    distance = abs(unit_pos[0] - nearest_enemy[0]) + abs(unit_pos[1] - nearest_enemy[1])

    if distance <= 4:
        # Attack enemy if within range
        return (5, nearest_enemy[0] - unit_pos[0], nearest_enemy[1] - unit_pos[1]), distance
    else:
        # Move toward the enemy
        return direction_to(unit_pos, nearest_enemy), distance
    
    
# MOVE TOWARD
def move_toward_target(unit_pos, target_pos):
    return direction_to(unit_pos, target_pos)



