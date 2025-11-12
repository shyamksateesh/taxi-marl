import gymnasium as gym
import time
import numpy as np

# 1. Initialize the Environment
# The standard, deterministic single-agent Taxi-v3 environment.
# 'ansi' mode is used for text-based rendering in the console.
env = gym.make("Taxi-v3", render_mode="ansi")

print("--- Environment Initialized ---")

# Reset the environment to get the initial state and info
# observation: The initial state of the environment (an integer from 0 to 499)
# info: Auxiliary information (empty dictionary for Taxi-v3 by default)
observation, info = env.reset()

print(f"Initial Observation (State Index): {observation}")

# 2. Analyze the Environment Spaces
# The state space is discrete (500 states) and the action space is discrete (6 actions)
print(f"Observation Space size: {env.observation_space.n} (500 possible states)") # [cite: 26]
print(f"Action Space size: {env.action_space.n} (6 possible actions)")       # 

# Action mapping (for reference):
# 0: south, 1: north, 2: east, 3: west, 4: pickup, 5: dropoff 
action_meanings = {0: "south", 1: "north", 2: "east", 3: "west", 4: "pickup", 5: "dropoff"}

# 3. Step Through the Environment
# We will run for a few steps, taking random actions

MAX_STEPS = 5
total_reward = 0

print("\n--- Running Simulation ---")

for step in range(MAX_STEPS):
    # Choose a random action from the action space
    action = env.action_space.sample()

    # Take a step in the environment
    # next_observation: New state index
    # reward: Reward received after taking the action (+20, -1, or -10) 
    # terminated: True if the episode ended (e.g., successful dropoff)
    # truncated: True if the episode ended due to a time limit (not relevant here)
    # info: Auxiliary information
    next_observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(f"\nStep {step + 1}:")
    print(f"  Action taken: {action} ({action_meanings[action]})")
    print(f"  Reward: {reward}")
    print(f"  New Observation (State Index): {next_observation}")

    # Render the environment's current state to the console
    print("  Current State Render:")
    print(env.render())

    if terminated or truncated:
        print("Episode finished.")
        break

    time.sleep(0.5) # Pause to see the steps

# 4. Cleanup
env.close()

print(f"--- Simulation Complete ---")
print(f"Total reward received: {total_reward}")