import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import random
import csv
import time
import matplotlib.pyplot as plt
from collections import deque

# Constants
NUM_EPOCHS = 50
CAPACITY = 100                # kWh
CHARGE_RATE = 10              # kW
DISCHARGE_RATE = 10           # kW
MIN_SOC = 0.3                 # 30% minimum (0-1 scale)
EPSILON_START = 1.0
EPSILON_MIN = 0.1
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 50000
UPDATE_TARGET_EVERY = 100
LEARNING_RATE = 0.0005

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = tf.convert_to_tensor([x[0].reshape(-1) for x in batch], dtype=tf.float32)
        actions = tf.convert_to_tensor([x[1] for x in batch], dtype=tf.int32)
        rewards = tf.convert_to_tensor([x[2] for x in batch], dtype=tf.float32)
        next_states = tf.convert_to_tensor([x[3].reshape(-1) for x in batch], dtype=tf.float32)
        dones = tf.convert_to_tensor([x[4] for x in batch], dtype=tf.bool)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.update_target_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(self.state_size,), activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    @tf.function
    def _train_step(self, states, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            loss = self.loss_fn(targets, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def predict(self, state):
        return self.model(tf.convert_to_tensor(state, dtype=tf.float32))
    
    def train(self, states, targets):
        self._train_step(states, targets)

class BatteryEnv:
    def __init__(self, data):
        self.data = data.sort_values(['Month', 'Day', 'Hour']).copy()
        self.data['DayID'] = self.data['Month'].astype(str) + '-' + self.data['Day'].astype(str)
        self.unique_days = self.data['DayID'].unique()
        self.current_day_idx = 0
        self.soc_carryover = 0.5  # 50% initial (0-1 scale)
        self.last_car_available = True
        self.prev_day_last_car_available = False
        self.min_soc = MIN_SOC
        self.capacity = CAPACITY
        self.charge_rate = CHARGE_RATE / CAPACITY  # Convert kW to kWh scale
        self.discharge_rate = DISCHARGE_RATE / CAPACITY
        self.max_grid_price = data['Grid_Price'].max()
        self.max_agg_price = data['Aggregator_Price'].max()
        self.max_demand = data['Household_Demand'].max()

    def reset_to_epoch_start(self):
        self.current_day_idx = 0
        self.soc_carryover = 0.5
        self.last_car_available = True
        self.prev_day_last_car_available = False

    def reset_day(self):
        actual_idx = self.current_day_idx % len(self.unique_days)
        current_day = self.unique_days[actual_idx]
        self.day_data = self.data[self.data['DayID'] == current_day].reset_index(drop=True)
        first_hour_available = self.day_data.iloc[0]['Car_Available']

        # Improved SOC initialization
        if first_hour_available:
            if self.prev_day_last_car_available:
                self.current_soc = self.soc_carryover
            else:
                first_valid_soc = self.day_data['SoC'].first_valid_index()
                if first_valid_soc is not None and not pd.isna(self.day_data.at[first_valid_soc, 'SoC']):
                    raw_soc = self.day_data.at[first_valid_soc, 'SoC']
                    self.current_soc = np.clip(raw_soc, 0.0, 1.0)
                else:
                    self.current_soc = self.soc_carryover
        else:
            self.current_soc = self.soc_carryover

        epoch_num = (self.current_day_idx // len(self.unique_days)) + 1
        print(f"\nEpoch {epoch_num}/{NUM_EPOCHS} | Day {actual_idx + 1}/{len(self.unique_days)} | SOC: {self.current_soc:.2f}")
        
        self.current_day_idx += 1
        self.current_idx = 0
        return self._get_state()

    def _get_state(self):
        row = self.day_data.iloc[self.current_idx]
        return np.array([[
            row['Hour'] / 24,
            row['Grid_Price'] / self.max_grid_price,
            row['Aggregator_Price'] / self.max_agg_price,
            row['Household_Demand'] / self.max_demand,
            self.current_soc
        ]], dtype=np.float32)

    def step(self, action):
        row = self.day_data.iloc[self.current_idx]
        current_car_available = row['Car_Available']
        demand = row['Household_Demand']
        battery_used = 0
        grid_used = 0
        cost = 0

        # Handle SOC transitions
        if current_car_available and not self.last_car_available:
            self.current_soc = np.clip(row['SoC'], self.min_soc, 1.0) if not pd.isna(row['SoC']) else self.soc_carryover

        if current_car_available:
            if action == 0:  # Charge
                charge_amount = min(self.charge_rate, 1.0 - self.current_soc)
                self.current_soc += charge_amount
                grid_used = demand
                cost = (charge_amount * self.capacity * row['Aggregator_Price']) + (grid_used * row['Grid_Price'])

            elif action == 1:  # Discharge
                available_energy = self.current_soc - self.min_soc
                discharge_amount = min(self.discharge_rate, demand / self.capacity, available_energy)
                self.current_soc -= discharge_amount
                battery_used = discharge_amount * self.capacity
                grid_used = max(demand - battery_used, 0)
                cost = grid_used * row['Grid_Price']

            elif action == 2:  # Idle
                grid_used = demand
                cost = grid_used * row['Grid_Price']
        else:
            grid_used = demand
            cost = grid_used * row['Grid_Price']
            action = 2  # Force idle

        # Calculate rewards
        baseline_cost = demand * row['Grid_Price']
        reward = (baseline_cost - cost)  # Direct savings incentive
        
        if current_car_available and action == 0:  
            if row['Aggregator_Price'] < 0.2: 
                norm_agg_price = row['Aggregator_Price'] / self.max_agg_price
                charge_bonus = (1 - norm_agg_price) * 15  
                reward += charge_bonus 

        # Penalties
        unmet_demand = max(demand - (grid_used + battery_used), 0)
        reward -= unmet_demand * grid_used * 10
        
        if self.current_soc < self.min_soc and action != 0:
            reward -= 50

        self.last_car_available = current_car_available
        self.current_idx += 1
        done = self.current_idx >= len(self.day_data)
        
        if done:
            self.prev_day_last_car_available = self.last_car_available
            if current_car_available:
                self.soc_carryover = self.current_soc
        
        next_state = np.zeros((1,5), dtype=np.float32) if done else self._get_state()
        return next_state, reward, done, {
            'grid_used': grid_used,
            'battery_used': battery_used,
            'total_cost': cost,
            'baseline_cost': baseline_cost,
            'soc': self.current_soc
        }

    def get_valid_actions(self):
        row = self.day_data.iloc[self.current_idx]
        valid_actions = [2]
        
        if row['Car_Available']:
            if self.current_soc < 1.0 - 1e-5:
                valid_actions.append(0)
            if self.current_soc > self.min_soc and row['Household_Demand'] > 0:
                valid_actions.append(1)
        return valid_actions

def train_dqn(train_path):
    start_time = time.time()
    df = pd.read_csv(train_path)
    env = BatteryEnv(df)
    state_size = 5
    action_size = 3
    
    agent = DQN(state_size, action_size)
    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    total_episodes = NUM_EPOCHS * len(env.unique_days)
    EPSILON_DECAY = (EPSILON_START - EPSILON_MIN) / (0.8 * total_episodes)
    epsilon = EPSILON_START
    all_rewards = []
    moving_avg_rewards = []
    
    with open('training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Day', 'Hour', 'Car_Available', 'Action',
                        'Initial_SoC', 'Final_SoC', 'Grid_Used', 'Battery_Used',
                        'Total_Cost', 'Reward', 'SOC_Carryover'])
        
        for episode in range(1, total_episodes + 1):
            state = env.reset_day()
            total_reward = 0
            done = False
            
            while not done:
                valid_actions = env.get_valid_actions()
                
                if np.random.rand() <= epsilon:
                    action = random.choice(valid_actions)
                else:
                    q_values = agent.predict(state).numpy()
                    valid_q_values = [q_values[0][a] for a in valid_actions]
                    action = valid_actions[np.argmax(valid_q_values)]
                
                next_state, reward, done, info = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)
                
                current_data = env.day_data.iloc[env.current_idx - 1]
                writer.writerow([
                    (episode-1)//len(env.unique_days) + 1,
                    (episode-1) % len(env.unique_days) + 1,
                    current_data['Hour'],
                    current_data['Car_Available'],
                    action,
                    float(state[0][4]),
                    info['soc'],
                    info['grid_used'],
                    info['battery_used'],
                    info['total_cost'],
                    reward,
                    env.soc_carryover
                ])
                
                total_reward += reward
                state = next_state
                
                if len(replay_buffer) >= BATCH_SIZE:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                    
                    # Double DQN with pure TensorFlow operations
                    current_q = agent.model(next_states)
                    max_actions = tf.cast(tf.argmax(current_q, axis=1), tf.int32)
                    target_q = agent.target_model(next_states)
                    max_next_q = tf.gather_nd(target_q, tf.stack([tf.range(BATCH_SIZE), max_actions], axis=1))
                    
                    targets = agent.model(states)
                    updates = rewards + GAMMA * max_next_q * tf.cast(~dones, tf.float32)
                    indices = tf.stack([tf.range(BATCH_SIZE), actions], axis=1)
                    targets = tf.tensor_scatter_nd_update(targets, indices, updates)
                    
                    agent.train(states, targets)
            
            all_rewards.append(total_reward)
            window_size = 20
            if episode >= window_size:
                moving_avg = np.mean(list(all_rewards)[-window_size:])
                moving_avg_rewards.append(moving_avg)
            
            epsilon = max(EPSILON_MIN, EPSILON_START - episode * EPSILON_DECAY)
            
            if episode % UPDATE_TARGET_EVERY == 0:
                agent.update_target_model()
            
            print(f"Epoch {((episode-1)//len(env.unique_days))+1}/{NUM_EPOCHS} | "
                  f"Episode {episode}/{total_episodes} | "
                  f"Reward: {total_reward:.2f} | "
                  f"ε: {epsilon:.3f}")

    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards, alpha=0.4, label='Episode Reward')
    if len(moving_avg_rewards) > 0:
        # Adjust the x-axis to start at window_size - 1 and have the same length as moving_avg_rewards
        plt.plot(range(window_size - 1, len(all_rewards)), moving_avg_rewards, 'r-', linewidth=2, label=f'{window_size}-Episode Moving Avg')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.title(f'Training Progress (Epochs={NUM_EPOCHS})', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_rewards.png', dpi=300)
    plt.show()

    total_time = time.time() - start_time
    agent.model.save_weights('trained_model_weights.h5')
    print(f"\nTraining completed in {total_time:.2f} seconds over {NUM_EPOCHS} epochs.")
    print(f"Total episodes: {total_episodes} | Average time per episode: {total_time/total_episodes:.2f}s")

def test_dqn(test_path):
    start_time = time.time()
    df = pd.read_csv(test_path)
    env = BatteryEnv(df)
    
    agent = DQN(5, 3)
    agent.model.load_weights('trained_model_weights.h5')
    
    monthly_results = {}
    
    with open('test_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Month', 'Day', 'Hour', 'Grid_Price', 'Aggregator_Price', 'Car_Available', 'Action',
                        'Initial_SoC', 'Final_SoC', 'Grid_Used', 'Battery_Used',
                        'AI_Cost', 'Baseline_Cost', 'Savings', 'Reward', 'SOC_Carryover'])
        
        for episode in range(1, len(env.unique_days) + 1):
            state = env.reset_day()
            done = False
            daily_ai = 0
            daily_base = 0
            month = None
            
            while not done:
                valid_actions = env.get_valid_actions()
                q_values = agent.predict(state).numpy()
                valid_q_values = [q_values[0][a] for a in valid_actions]
                action = valid_actions[np.argmax(valid_q_values)]
                
                next_state, reward, done, info = env.step(action)
                current_data = env.day_data.iloc[env.current_idx - 1]
                
                if month is None:
                    month = current_data['Month']
                
                hourly_savings = info['baseline_cost'] - info['total_cost']
                
                writer.writerow([
                    month,
                    current_data['Day'],
                    current_data['Hour'],
                    current_data['Grid_Price'],
                    current_data['Aggregator_Price'], 
                    current_data['Car_Available'],
                    action,
                    float(state[0][4]),
                    info['soc'],
                    info['grid_used'],
                    info['battery_used'],
                    info['total_cost'],
                    info['baseline_cost'],
                    hourly_savings,
                    reward,
                    env.soc_carryover
                ])
                
                daily_ai += info['total_cost']
                daily_base += info['baseline_cost']
                state = next_state
            
            # Update monthly totals
            if month not in monthly_results:
                monthly_results[month] = {
                    'ai_cost': 0,
                    'baseline_cost': 0,
                    'days': 0
                }
            monthly_results[month]['ai_cost'] += daily_ai
            monthly_results[month]['baseline_cost'] += daily_base
            monthly_results[month]['days'] += 1
            
            print(f"Day {episode} ({month}): AI: ${daily_ai:.2f} | Baseline: ${daily_base:.2f} | Saved: ${daily_base - daily_ai:.2f}")

    # Monthly reporting
    print("\n=== Monthly Test Results ===")
    for month in sorted(monthly_results.keys()):
        data = monthly_results[month]
        total_savings = data['baseline_cost'] - data['ai_cost']
        savings_pct = (total_savings / data['baseline_cost']) * 100 if data['baseline_cost'] > 0 else 0
        
        print(f"\nMonth {month}:")
        print(f"- Days Analyzed: {data['days']}")
        print(f"- Total AI Cost: ${data['ai_cost']:.2f}")
        print(f"- Total Baseline Cost: ${data['baseline_cost']:.2f}")
        print(f"- Total Savings: ${total_savings:.2f}")
        print(f"- Savings Percentage: {savings_pct:.1f}%")
        print(f"- Average Daily Savings: ${total_savings/data['days']:.2f}")

    # Visualization
    months = sorted(monthly_results.keys())
    ai_costs = [monthly_results[m]['ai_cost'] for m in months]
    baseline_costs = [monthly_results[m]['baseline_cost'] for m in months]
    savings = [b - a for a,b in zip(ai_costs, baseline_costs)]

    plt.figure(figsize=(15, 6))
    
    # Cost Comparison Plot
    plt.subplot(1, 2, 1)
    bar_width = 0.35
    x_indices = np.arange(len(months))
    
    plt.bar(x_indices - bar_width/2, ai_costs, width=bar_width,
            color='#1f77b4', label='AI Cost')
    plt.bar(x_indices + bar_width/2, baseline_costs, width=bar_width,
            color='#ff7f0e', label='Baseline Cost')
    
    plt.xlabel('Month', fontsize=20)
    plt.ylabel('Total Cost ($)', fontsize=20)
    plt.title('Monthly Cost Comparison', fontsize=24)
    plt.xticks(x_indices, months)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Savings Plot
    plt.subplot(1, 2, 2)
    plt.bar(months, savings, color='#2ca02c')
    plt.xlabel('Month', fontsize=20)
    plt.ylabel('Total Savings ($)', fontsize=20)
    plt.title('Monthly Savings', fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('monthly_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    total_time = time.time() - start_time
    print(f"\nTesting completed in {total_time:.2f} seconds.")
    return monthly_results



if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    #train_dqn("battery_dataset.csv")
    test_dqn("test_dataset.csv")
       