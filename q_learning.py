import numpy as np
import pygame
import time
import sys
import signal
import os

NUM_EPISODES = 9999
TICKS_PER_SECOND = 100
LEARNING_RATE = 0.2
DISCOUNT_RATE = 0.9
EXPLORATION_RATE = 1.0
EXPLORTATION_RATE_DECAY = 0.9998

UP = "\x1B[3A"
CLR = "\x1B[0K"

os.system('clear')

class GridEnvironment:
    def __init__(self, rows, cols, start, goal, obstacles=[]):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.agent_position = start

    def reset(self):
        self.agent_position = self.start
        return self.agent_position

    def is_adjacent(self, position, current_position):
        row_diff = abs(position[0] - current_position[0])
        col_diff = abs(position[1] - current_position[1])
        return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)

    def is_valid_position(self, position, current_position):
        valid = (
            0 <= position[0] < self.rows
            and 0 <= position[1] < self.cols
            and position not in self.obstacles
            and self.is_adjacent(position, current_position)
        )
        return valid

    def get_possible_actions(self, state):
        possible_actions = ['up', 'down', 'left', 'right']
        current_row, current_col = state

        if current_row == 0:
            possible_actions.remove('up')
        if current_row == self.rows - 1:
            possible_actions.remove('down')
        if current_col == 0:
            possible_actions.remove('left')
        if current_col == self.cols - 1:
            possible_actions.remove('right')

        for obstacle_row, obstacle_col in self.obstacles:
            if (obstacle_row, obstacle_col) == (current_row - 1, current_col):
                possible_actions.remove('up')
            elif (obstacle_row, obstacle_col) == (current_row + 1, current_col):
                possible_actions.remove('down')
            elif (obstacle_row, obstacle_col) == (current_row, current_col - 1):
                possible_actions.remove('left')
            elif (obstacle_row, obstacle_col) == (current_row, current_col + 1):
                possible_actions.remove('right')

        return possible_actions

    def take_action(self, action):
        actions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        new_position = tuple(np.add(self.agent_position, actions[action]))

        if self.is_valid_position(new_position, self.agent_position):
            self.agent_position = new_position
            
            if self.agent_position == self.goal:
                reward = 1
                done = True
            else:
                reward = 0
                done = False
        else:
            reward = 0
            done = False

        return self.agent_position, reward, done

class QLearningAgent:
    def __init__(self, num_actions, initial_learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_RATE,
                 initial_exploration_rate=EXPLORATION_RATE, exploration_decay=EXPLORTATION_RATE_DECAY):
        self.num_actions = num_actions
        self.learning_rate = initial_learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = initial_exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), np.random.normal(0, 0.01))

    def choose_action(self, state, possible_actions):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(possible_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in possible_actions]
            return possible_actions[np.argmax(q_values)]

    def update_q_value(self, state, action, next_state, reward, done):
        current_q = self.get_q_value(state, action)

        if done:
            new_q = reward
        else:
            max_next_q = max([self.get_q_value(next_state, next_action) for next_action in env.get_possible_actions(next_state)])
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

        self.q_table[(state, action)] = new_q

        self.exploration_rate *= self.exploration_decay

def handle_interrupt(signal, frame):
    show_cursor()
    sys.exit(0)
    
signal.signal(signal.SIGINT, handle_interrupt)  
  
def hide_cursor():
    
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

def show_cursor():
    os.system('clear')
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()

def update_progress(exp_rate, moves_total, moves_ep, time, ep):
    line1 = f'exploration rate    total moves    ep moves    ep time   episode'
    line2 = f'{exp_rate:.5f}'.ljust(20) + f'{moves_total}'.ljust(15) + f'{moves_ep}'.ljust(12) + f'{time:.3f}'.ljust(10) + f'{ep}'.ljust(6)
    print(f'{UP}{line1}{CLR}\n{line2}{CLR}\n')

def draw_grid(agent_position):
    window.fill((255, 255, 255))

    for row in range(rows):
        for col in range(cols):
            pygame.draw.rect(window, (255, 255, 255), (col * cell_size, row * cell_size, cell_size, cell_size), 1)

    for obstacle in obstacles:
        pygame.draw.rect(window, (0, 0, 0), (obstacle[1] * cell_size, obstacle[0] * cell_size, cell_size, cell_size))

    pygame.draw.rect(window, (255, 0, 0), (agent_position[1] * cell_size, agent_position[0] * cell_size, cell_size, cell_size))
    pygame.draw.rect(window, (0, 255, 0), (goal_state[1] * cell_size, goal_state[0] * cell_size, cell_size, cell_size))
   
if __name__ == "__main__":
    pygame.init()
    hide_cursor()

    rows, cols = 10, 10
    start_state = (0, 0)
    goal_state = (9, 9)
    obstacles = [(1,1), (1,2), (1,3), (1,4), (0,6),
                (1,6), (1,8), (2,8), (3,8), (5,8),
                (6,8), (7,8), (8,8), (8,9), (8,7),
                (8,6), (6,6), (5,6), (4,6), (3,6),
                (2,1), (3,1), (4,1), (5,1), (6,1),
                (7,1), (8,1), (8,2), (3,4), (3,3),
                (4,3), (5,3), (6,3), (6,4), (7,4),
                (8,4), (9,4)]

    env = GridEnvironment(rows, cols, start_state, goal_state, obstacles)
    agent = QLearningAgent(num_actions=len(env.get_possible_actions(start_state)))

    cell_size = 50
    grid_width, grid_height = cols * cell_size, rows * cell_size

    window_width, window_height = grid_width, grid_height
    window = pygame.display.set_mode((window_width, window_height))

    clock = pygame.time.Clock()

    running = True

    elapsed_time = 0.0
    moves_total = 0
    
    for episode in range(NUM_EPISODES):
        moves_per_episode = 0
        start_time = time.time()
        
        state = env.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            possible_actions = env.get_possible_actions(state)
            action = agent.choose_action(state, possible_actions)

            next_state, reward, done = env.take_action(action)
            
            moves_total += 1
            moves_per_episode += 1

            agent.update_q_value(state, action, next_state, reward, done)

            draw_grid(env.agent_position)
            pygame.display.flip()
            clock.tick(TICKS_PER_SECOND)

            state = next_state
            
            update_progress(agent.exploration_rate, moves_total, moves_per_episode, elapsed_time, episode)

        end_time = time.time()
        elapsed_time = end_time - start_time
        
    show_cursor()

    pygame.quit()
    sys.exit()
