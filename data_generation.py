import numpy as np
import pickle

class GenerateData:
    def generate_training_data(num_samples=1000, grid_size=(10, 10)):
        grids = []
        positions = []
        goals = []
        heuristics = []

        for _ in range(num_samples):
            grid = np.zeros(grid_size)
            for _ in range(np.random.randint(20, 40)):
                x, y = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
                grid[x, y] = 1
            
            start_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            goal_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))

            while grid[start_pos[0], start_pos[1]] == 1:
                start_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            while grid[goal_pos[0], goal_pos[1]] == 1:
                goal_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))

            h_manhattan = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])

            grids.append(grid.flatten())
            positions.append(start_pos)
            goals.append(goal_pos)
            heuristics.append(h_manhattan)

        grids = np.array(grids)
        positions = np.array(positions)
        goals = np.array(goals)
        heuristics = np.array(heuristics)
        


        inputs = np.hstack([grids, positions, goals])
        return inputs, heuristics
    def generate_training_data(num_samples=1000, grid_size=(10, 10), blocked_blocks_range=(5, 20)):
        grids = []
        positions = []
        goals = []
        heuristics = []

        for _ in range(num_samples):
            grid = np.zeros(grid_size)
            for _ in range(np.random.randint(blocked_blocks_range[0], blocked_blocks_range[1])):
                x, y = np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1])
                grid[x, y] = 1
            
            start_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            goal_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))

            while grid[start_pos[0], start_pos[1]] == 1:
                start_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            while grid[goal_pos[0], goal_pos[1]] == 1:
                goal_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))

            h_manhattan = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])

            grids.append(grid.flatten())
            positions.append(start_pos)
            goals.append(goal_pos)
            heuristics.append(h_manhattan)

        grids = np.array(grids)
        positions = np.array(positions)
        goals = np.array(goals)
        heuristics = np.array(heuristics)
        


        inputs = np.hstack([grids, positions, goals])
        return inputs, heuristics
    
    
    def generate_training_data_from_images(images):
        inputs = []
        heuristics = []

        for grid in images:
            grid = np.array(grid)
            grid_size = grid.shape
            start_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            while grid[start_pos[0], start_pos[1]] == 1:
                start_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            goal_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            while grid[goal_pos[0], goal_pos[1]] == 1 or goal_pos == start_pos:
                goal_pos = (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))

            h_manhattan = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1])

            flattened_grid = grid.flatten()
            input_row = np.hstack([flattened_grid, start_pos, goal_pos])

            inputs.append(input_row)
            heuristics.append(h_manhattan)

        inputs = np.array(inputs)
        heuristics = np.array(heuristics)

        return inputs, heuristics
    
    def save_training_data_as_pkl(data, name):
        with open('training_data/'+name+'.pkl', 'wb') as file:  
            pickle.dump(data, file)
            
    def load_training_data_from_pkl(name):
        with open('training_data/'+name+'.pkl', 'rb') as file:  
            data = pickle.load(file)
        return data
    
