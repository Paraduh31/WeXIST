import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from Trading_Environment import StockTradingEnv  # Import your environment
import yfinance as yf


# Download data
ticker = "AAPL"
df = yf.download(ticker, start="2023-01-01", end="2023-04-01", progress=False)

# Check the MultiIndex structure
print(df.columns)

# Flatten the MultiIndex
df.columns = [' '.join(col).strip() for col in df.columns.values]

# Check the new columns
print(df.columns)

# Now access the 'close' price for 'AAPL'
df['Close AAPL']  # Accessing 'close' price for 'AAPL'
df['SMA'] = df['Close AAPL'].rolling(window=10).mean()  # Calculate SMA using the 'close' price
df['SMA'] = df['SMA'].fillna(method='bfill')

# Convert to the required format
data = {
    'date': df.index.tolist(),
    'close': df['Close AAPL'].tolist(),
    'SMA': df['SMA'].tolist(),
}
df= pd.DataFrame(data)

# Initialize the custom StockTrading environment
env = StockTradingEnv(
    df=df,
    stock_dim=1,  # Single stock
    hmax=10,  # Max shares that can be traded per step
    initial_amount=10000,  # Initial cash balance
    num_stock_shares=[0],  # Start with 0 shares
    buy_cost_pct=[0.01],  # 1% buy cost
    sell_cost_pct=[0.01],  # 1% sell cost
    reward_scaling=1.0,  # Reward scaling factor
    state_space=4,  # Number of elements in state representation
    tech_indicator_list=['SMA'],  # Include SMA as a technical indicator
    make_plots=False,
    print_verbosity=10
)

# Wrap environment in DummyVecEnv for Stable Baselines compatibility
env = DummyVecEnv([lambda: env])

# Reset environment
obs = env.reset()

# Take 20 random actions with random magnitudes between 0 and 1000
for _ in range(10):
    direction = np.random.choice([-1, 1])  # -1 for sell, 1 for buy
    magnitude = np.random.uniform(0, 1)  # Ensure magnitude is between 0 and 1
    action = np.array([direction, magnitude], dtype=np.float32).reshape(1, -1)  # Reshape for DummyVecEnv

    obs, reward, done, truncated= env.step(action)
    print(f"Action Taken: {action}, Reward: {reward}, Done: {done} ")

    if done:
        print("Episode finished. Resetting environment.")
        obs = env.reset()



# Close the environment after use
env.close()
