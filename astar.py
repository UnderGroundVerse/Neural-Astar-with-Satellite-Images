import numpy as np
import math
import matplotlib.pyplot as plt


class Node:
    def __init__(self, position, g=0, h=0, f=0, parent_node=None):
        self.position = position
        self.g = g
        self.h = h
        self.f = f
        self.parent_node = parent_node
        self.path = []

    def __eq__(self, other):
        return isinstance(other, Node) and self.position == other.position

    def __hash__(self):
        return hash(self.position)

class Astar:
    def __init__(self, grid: np.ndarray, goal_position, starting_position=(0,0), h_type='manhattan'):
        self.grid = grid
        self.goal_position = goal_position
        self.open_list = [Node(starting_position, 0,0)]
        self.closed_list = []
        self.h_type = h_type.lower()
        self.path=[]

    def is_position_valid(self, position):
        if position[0] > self.grid.shape[0] - 1 or position[1] > self.grid.shape[1] - 1: 
            return False
        if position[0] < 0 or position[1] < 0:
            return False
        if self.is_position_blocked(position):
            return False
        return True

    def is_position_blocked(self, position):
        return int(self.grid[int(position[0]), int(position[1])]) == 1

    def calc_h(self, node):
        if self.h_type == 'manhattan':
            result = abs(node.position[0] - self.goal_position[0]) + abs(node.position[1] - self.goal_position[1])
        elif self.h_type == 'euclidean':
            result = math.sqrt((node.position[0] - self.goal_position[0])**2 + (node.position[1] - self.goal_position[1])**2)
        else:
            raise ValueError('H_type is not defined')
        node.h = result
        return result

    def calc_g(self, node):
        node.g = node.parent_node.g + 1 
        return node.g

    def calc_f(self, node):
        node.f = node.g + node.h
        return node.f

    def get_neighbors(self, position):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        result = []

        for direction in directions:
            x, y = position[0] + direction[0], position[1] + direction[1]
            if self.is_position_valid((x, y)):
                result.append((x, y))

        return result

    def find_path(self):
        while self.open_list:
            current_node = min(self.open_list, key=lambda x: x.f)

            if current_node.position[0] == self.goal_position[0] and current_node.position[1] == self.goal_position[1]:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent_node
                self.path = path[::-1]
                return path[::-1]  # Return the path here

            self.open_list.remove(current_node)
            self.closed_list.append(current_node)

            for neighbor_pos in self.get_neighbors(current_node.position):
                neighbor_node = Node(neighbor_pos, parent_node=current_node)

                if neighbor_node in self.closed_list:
                    continue

                self.calc_g(neighbor_node)
                self.calc_h(neighbor_node)
                self.calc_f(neighbor_node)

                # Check if the neighbor is already in the open list
                in_open_list = False
                for open_node in self.open_list:
                    if neighbor_node.position[0] == open_node.position[0] and neighbor_node.position[1] == open_node.position[1]   and neighbor_node.g >= open_node.g:
                        in_open_list = True
                        break

                if not in_open_list:
                    self.open_list.append(neighbor_node)

        return None  # No path found
    
    def visualize_path(self, path=None):
        path = self.path if path==None else path 
        
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary')
        
        if path:
            path = np.array(path)
            plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=3, label='Path')
            plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
            plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='Goal')
        
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.title("A* Pathfinding Result")
        plt.show()
        
        
        
class DifferentialAstar(Astar):
    def __init__(self, grid, heuristic_model, goal_position, starting_position=(0, 0), h_type='manhattan'):
        super().__init__(grid, goal_position, starting_position, h_type)
        self.heuristic_model = heuristic_model
        
    def calc_h(self, node):
        grid_flat = self.grid.flatten()
        input_data = np.hstack([grid_flat, node.position, self.goal_position]).reshape(1, -1)
        node.h = self.heuristic_model.predict(input_data)[0, 0]
        return node.h
        
        
        
if __name__ == '__main__':
    grid = np.zeros((20, 20))
    grid[5:15, 10] = 1  # Vertical wall
    grid[5, 5:15] = 1   # Horizontal wall
    start_pos = (2, 2)
    goal_pos = (10, 18)
    
    astar = Astar(grid, goal_position=goal_pos, starting_position=start_pos)
    astar.find_path()
    astar.visualize_path()