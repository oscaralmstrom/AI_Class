import re
from tracemalloc import start
from typing import List, Tuple, Set, Dict, Optional, cast
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmState
from heapq import heappush, heappop
import time


class Node:
    def __init__(self, state: State, path_cost: float, parent_action: Optional[int], parent, depth):
        self.state: State = state
        self.parent: Optional[Node] = parent
        self.path_cost: float = path_cost
        self.parent_action: Optional[int] = parent_action
        self.depth: int = depth

    def __hash__(self):
        return self.state.__hash__()

    def __gt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


def get_next_state_and_transition_cost(env: Environment, state: State, action: int) -> Tuple[State, float]:
    rw, states_a, _ = env.state_action_dynamics(state, action)
    state: State = states_a[0]
    transition_cost: float = -rw

    return state, transition_cost


def visualize_bfs(viz, closed_set: Set[State], queue: List[Node], wait: float):
    grid_dim_x, grid_dim_y = viz.env.grid_shape
    for pos_i in range(grid_dim_x):
        for pos_j in range(grid_dim_y):
            viz.board.itemconfigure(viz.grid_squares[pos_i][pos_j], fill="white")

    for state_u in closed_set:
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")

    for node in queue:
        state_u: FarmState = cast(FarmState, node.state)
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="grey")

    viz.window.update()
    time.sleep(wait)


def visualize_dfs(viz, popped_node: Node, lifo: List[Node]):
    grid_dim_x, grid_dim_y = viz.env.grid_shape
    for pos_i in range(grid_dim_x):
        for pos_j in range(grid_dim_y):
            viz.board.itemconfigure(viz.grid_squares[pos_i][pos_j], fill="white")

    for node in lifo:
        state_u: FarmState = cast(FarmState, node.state)
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="grey")

    node_parent = popped_node.parent
    while node_parent is not None:
        parent_state_u: FarmState = cast(FarmState, node_parent.state)
        pos_i_up, pos_j_up = parent_state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")
        node_parent = node_parent.parent

    viz.window.update()


def breadth_first_search(start_state: State, env: Environment, viz) -> Optional[List[int]]:
    """ Breadth-first search

    :param start_state: starting state
    :param env: environment
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """
    #Terminate if the start state is the goal
    if env.is_terminal(start_state): 
        return []
    
    #Init the open fifo list and closed set
    open = list()
    closed = set()
    root = Node(start_state, 0, None, None, 0)
    open.append(root)
    closed.add(root.state)

    while len(open) > 0:
        #As its a fifo queue, always pop the first element
        parent = open.pop(0)

        #Get available actions this parent node can do
        for action in env.get_actions(parent.state):

            #Create new child node based on an action taken from parent node
            [new_state, cost] = get_next_state_and_transition_cost(env, parent.state, action)
            child = Node(new_state, parent.path_cost + cost, action, parent, parent.depth + 1)

            #If the goal is found, return the solution
            if env.is_terminal(child.state):
                return get_solution(child, start_state)
            
            if child.state not in closed:
                #Append defaults to adding to back of list, fifo still holds
                closed.add(child.state)
                open.append(child)
    
    return None

#Returns a list of actions that resultes in a solution from node child to state start_state
def get_solution(child: Node, start_state: State) ->List[int]:
    result_actions = list()
    #Igonre the root's parent action as it is initialized to None
    while (child.state != start_state):
        result_actions.append(child.parent_action)
        #Continue following along the parent's previous action
        child = child.parent    
    #Reverse the list as we create it starting at the goal
    result_actions.reverse()
    return result_actions

def depth_limited_search(start_state: State, env: Environment, limit: int, viz) -> Optional[List[int]]:
    """ Depth-limited search

    :param start_state: starting state
    :param env: environment
    :param limit: depth-limit
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    was found
    """
   #Terminate if the start state is the goal
    if env.is_terminal(start_state): 
        return []
    
    #Init the open LIFO list and closed set
    open = list()
    closed = set()
    root = Node(start_state, 0, None, None, 0)
    open.append(root)
    closed.add(root.state)

    while len(open) > 0:
        #As its a LIFO queue, always pop the last (default) element
        parent = open.pop()

        if env.is_terminal(parent.state):
            return get_solution(parent, start_state)

        #Go to the next node in the LIFO if depth reach or cycle detected
        if (parent.depth >= limit or isCycle(parent, start_state)):
            continue

        #Get available actions this parent node can do
        for action in env.get_actions(parent.state):

            #Create new child node based on an action taken from parent node
            [new_state, cost] = get_next_state_and_transition_cost(env, parent.state, action)
            child = Node(new_state, parent.path_cost + cost, action, parent, parent.depth + 1)

            #If the goal is found, return the solution
            if env.is_terminal(child.state):
                return get_solution(child, start_state)
            
            if child.state not in closed:
                #Append defaults by adding to the back of the list, LIFO still holds
                closed.add(child.state)
                open.append(child)
    
    return None

def isCycle(node: Node, start_state: State) -> bool:
    #Store all visited states to get to this node
    visited_states = set()
    #Will loop until a same previous state is detected (a cycle)
    while (node.state not in visited_states):
        visited_states.add(node.state)
        #If we have reached the start_state before any repeated state, then no cycle is present
        if (node.state == start_state):
             return False
        node = node.parent
    return True

def iterative_deepening_search(start_state: State, env: Environment, viz) -> List[int]:
    """ Iterative-deepening search

    :param start_state: starting state
    :param env: environment
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal
    """

    solution = None
    limit = 0
    
    while (solution == None):
        solution = depth_limited_search(start_state, env, limit, viz)
        limit += 1
    return solution


def best_first_search(start_state: State, env: Environment, weight_g: float, weight_h: float,
                      viz) -> Optional[List[int]]:
    """ Best-first search

    :param start_state: starting state
    :param env: environment
    :param weight_g: path cost weight
    :param weight_h: heuristic weight
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """

    #Terminate if the start state is the goal
    if env.is_terminal(start_state): 
        return []
    
    #Init the open fifo list and closed set
    open = []
    closed = dict()
    root = Node(start_state, 0, None, None, 0)
    heappush(open,(0, root))
    closed[root.state] = 0

    while len(open) > 0:
        #As its a fifo queue, always pop the first element
        (path_cost, parent) = heappop(open)

        if env.is_terminal(parent.state):
            return get_solution(parent, start_state)

        #Get available actions this parent node can do
        for action in env.get_actions(parent.state):

            #Create new child node based on an action taken from parent node
            [new_state, cost] = get_next_state_and_transition_cost(env, parent.state, action)
            child = Node(new_state, path_cost + cost, action, parent, parent.depth + 1)

            if (child.state not in closed or child.path_cost < closed[child.state]):
                #Append defaults to adding to back of list, fifo still holds
                closed[child.state] = child.path_cost
                heappush(open, (priorityCostFunction(child, weight_g, weight_h), child))
    
    return None

def priorityCostFunction(node: Node, weight_g: float, weight_h: float) -> float:
    (nX, nY) = node.state.agent_idx
    (gX, gY) = node.state.goal_idx
    manhattan = abs(nX - gX) + abs(nY - gY)
    return node.path_cost*(weight_g + weight_h*manhattan)

