import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.utils import seeding


class StockTradingEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df: pd.DataFrame,  # Stock market data
            stock_dim: int,  # Number of stocks in the environment
            hmax: int,  # Maximum number of shares that can be bought/sold in one step
            initial_amount: float,  # Initial cash balance
            num_stock_shares: list[int],  # Number of shares held for each stock
            buy_cost_pct: list[float],  # Buying transaction cost per stock (as a percentage)
            sell_cost_pct: list[float],  # Selling transaction cost per stock (as a percentage)
            reward_scaling: float,  # Scaling factor for rewards
            state_space: int,  # Number of elements in the state representation
            tech_indicator_list: list[str],  # List of technical indicators used as features
            make_plots: bool = False,  # Whether to generate plots
            print_verbosity=10,  # Frequency of printing logs
            day=0,  # Start day index for the dataset
            initial=True,  # Whether it's the initial state
            previous_state=None,  # Previous state (if resuming)
            model_name="",  # Name of the model (for logging)
            mode="",  # Mode of the environment (e.g., training/testing)
            iteration="",  # Current iteration number (for tracking)
    ):
        super(StockTradingEnv, self).__init__()

        # Log the initialization process
        print("🚀 Initializing StockTradingEnv...")

        # Assign parameters to class attributes
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.tech_indicator_list = tech_indicator_list
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state if previous_state is not None else []
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.episode = 0

        # Log dataset details
        print(f"📊 Loaded dataset with {len(self.df)} records and {self.stock_dim} stocks.")

        # Define Action Space
        # - First value: Buy (-1 to 1), Sell (-1 to 1) per stock
        # - Second value: Allocation percentage (0 to 1)
        self.action_space = spaces.Box(
            low=np.array([-1, 0]),  # Min: (-1 for selling, 0% allocation)
            high=np.array([1, 1]),  # Max: (1 for buying, 100% allocation)
            shape=(2,),  # 2 values per stock
            dtype=np.float32
        )

        # Define Observation Space (State Representation)
        # - This includes stock prices, owned shares, cash balance, and technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )

        # Log space details
        print(f"🎯 Defined action space: {self.action_space.shape}")
        print(f"🧐 Defined observation space: {self.observation_space.shape}")

        # Initialize the first state
        self.data = self.df.loc[self.day, :]
        self.terminal = False  # Indicates whether the episode has ended
        self.state = self._initiate_state()

        # Initialize tracking variables
        self.reset_tracking_variables()

        print("✅ StockTradingEnv initialization complete! Ready to trade! 📈")

    def reset_tracking_variables(self):
        """
        Reset all tracking variables for a new episode.
        This function is called at the start of each new episode to reset the environment.
        """

        print("\n[INFO] Resetting tracking variables...")

        # Reset rewards and costs
        self.reward = 0
        self.cost = 0
        self.trades = 0

        # Increment episode count
        self.episode  += 1
        print(f"[INFO] Starting Episode {self.episode}")

        # Reset asset tracking
        self.asset_memory = [self.initial_amount]
        print(f"[INFO] Initial cash balance: {self.initial_amount:.2f}")

        # Reset memory for tracking performance
        self.rewards_memory = []  # Stores reward history
        self.actions_memory = []  # Stores actions taken
        self.state_memory = []  # Stores state history
        self.date_memory = [self._get_date()]  # Stores dates of each step

        print("[INFO] Tracking variables reset complete.\n")

    def _initiate_state(self):
        print(f"🛠 [DEBUG] Initializing state...")

        # If it's the start of a new episode, use the initial values
        if self.initial:
            print("📌 [INFO] Using initial state values.")

            state = [
                self.initial_amount,  # Initial cash balance
                *self.num_stock_shares,  # Number of shares held for each stock
                self.data['close'],  # Stock close price (single value)
                *[self.data[tech] for tech in self.tech_indicator_list],  # Tech indicators (single values)
            ]

        else:
            print("📌 [INFO] Using previous state values.")

            state = [
                self.previous_state[0],  # Previous cash balance
                *self.previous_state[1:self.stock_dim + 1],  # Previous stock holdings
                self.data['close'],  # Current stock close price (single value)
                *[self.data[tech] for tech in self.tech_indicator_list],  # Tech indicators (single values)
            ]

        print(f"✅ [DEBUG] State initialized: {state}")
        return state

    def step(self, action):
        """
        Execute a single step in the trading environment.

        Process:
        1. Execute the given trading action (buy/sell/hold).
        2. Move to the next trading day.
        3. Update the state based on new market data.
        4. Calculate the current portfolio value.
        5. Compute the reward as the percentage change in portfolio value.
        6. Store important data for tracking and visualization.
        7. Return the updated state, reward, and whether the episode is done.

        Args:
            action (array): The action to execute, where:
                - action[0]: Buy/sell signal (-1 to 1)
                - action[1]: Allocation percentage (0 to 1)

        Returns:
            state (np.array): The updated state of the environment.
            reward (float): The reward for this step (change in portfolio value).
            terminal (bool): Whether the episode has ended.
            truncated (bool): Always False (no truncation logic).
            info (dict): Additional information like current date and portfolio value.
        """

        print("\n[INFO] Executing step in the environment...")
        print(f"[INFO] Current trading day: {self.day}")
        print(f"[INFO] Executing action: {action}")

        # Step 1: Execute the trading action (Buy/Sell/Hold)
        self._execute_action(action)

        # Step 2: Move to the next day
        self.day += 1
        print(f"[INFO] Moved to next trading day: {self.day}")

        # Step 3: Check if episode is complete
        if self.day >= len(self.df.index) - 1:
            self.terminal = True
            print("[INFO] End of dataset reached. Episode terminating.")

        # Step 4: Update the state with new day's market data (if not terminal)
        if not self.terminal:
            self.data = self.df.loc[self.day, :]
            self.state = self._update_state()
            print(f"[INFO] Updated state for day {self.day}")

        # Step 5: Calculate portfolio value
        current_portfolio_value = (
                self.state[0] +  # Cash balance
                np.sum(np.array(self.state[1:self.stock_dim + 1]) *  # Number of shares held
                       np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1]))  # Stock prices
        )
        print(f"[INFO] Portfolio Value: {current_portfolio_value:.2f}")

        # Step 6: Compute reward (Change in portfolio value)
        prev_portfolio_value = self.asset_memory[-1] if len(self.asset_memory) > 1 else self.initial_amount
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward *= self.reward_scaling  # Scale reward for better learning
        print(f"[INFO] Reward calculated: {reward:.6f}")

        # Step 7: Store data for memory tracking
        self.asset_memory.append(current_portfolio_value)
        self.rewards_memory.append(reward)
        self.date_memory.append(self._get_date())
        self.state_memory.append(self.state)
        self.actions_memory.append(action)

        # Step 8: Optional plotting if episode ends
        if self.terminal and self.make_plots:
            print("[INFO] Generating plot for portfolio performance...")
            self._make_plot()

        # Step 9: Reset cost and trade count for next step
        self.cost = 0
        self.trades = 0

        # Step 10: Return the new state, reward, and episode status
        return (
            np.array(self.state, dtype=np.float32),  # Updated state
            reward,  # Reward for this step
            self.terminal,  # Whether episode is finished
            False,  # Truncated (not used in this case)
            {"date": self._get_date(), "portfolio_value": current_portfolio_value}  # Additional info
        )

    def _execute_action(self, action):
        """
        Execute a trade based on the provided action.

        Action consists of two values:
        - direction (action[0]): Determines whether to Buy (>0), Sell (<0), or Hold (0).
        - magnitude (action[1]): Determines the percentage of available cash/shares to trade.

        This function updates:
        - Cash balance
        - Number of shares held
        - Transaction costs
        - Number of trades executed

        Args:
            action (list or np.array): A 2D action where:
                - action[0] (direction) is in range [-1, 1]
                - action[1] (magnitude) is in range [0, 1]
        """

        # Extract action components
        direction, magnitude = action[0], action[1]

        # Extract current environment state
        cash = self.state[0]  # Available cash
        shares = self.state[1]  # Shares currently held
        price = self.state[self.stock_dim + 1]  # Current stock price
        buy_cost = self.buy_cost_pct[0]  # Buy transaction fee percentage
        sell_cost = self.sell_cost_pct[0]  # Sell transaction fee percentage

        print("\n[INFO] Executing trade action...")
        print(f"[INFO] Action received: Direction = {direction:.2f}, Magnitude = {magnitude:.2f}")
        print(f"[INFO] Current state: Cash = {cash:.2f}, Shares = {shares}, Price = {price:.2f}")

        # Buy Action
        if direction > 0: # and magnitude > 0.01:
            print("[INFO] Attempting to BUY shares...")

            # Calculate investable amount based on available cash and magnitude
            investable_amount = cash * magnitude
            max_shares = np.floor(investable_amount / (price * (1 + buy_cost)))  # Max shares affordable
            buy_shares = min(max_shares, self.hmax)  # Cap shares to max allowed per step
            total_cost = buy_shares * price * (1 + buy_cost)  # Total transaction cost

            # Check if the purchase can proceed
            if total_cost <= cash:
                # Update cash and shares
                self.state[0] -= total_cost
                self.state[1] += buy_shares
                self.cost += total_cost - (buy_shares * price)  # Store cost incurred
                self.trades += 1 if buy_shares > 0 else 0  # Count as a trade if shares were bought

                print(f"[INFO] Bought {buy_shares} shares at {price:.2f}, total cost: {total_cost:.2f}")
            else:
                print("[WARNING] Not enough cash to execute buy order.")

        # Sell Action
        elif direction < 0: # and magnitude > 0.01:
            print("[INFO] Attempting to SELL shares...")

            # Determine the number of shares to sell based on magnitude
            sell_shares = np.floor(shares * magnitude)
            sell_shares = min(sell_shares, self.hmax)  # Cap shares to max allowed per step
            revenue = sell_shares * price * (1 - sell_cost)  # Total revenue after transaction fee

            # Check if shares are available to sell
            if sell_shares > 0:
                # Update cash and shares
                self.state[0] += revenue
                self.state[1] -= sell_shares
                self.cost += sell_shares * price * sell_cost  # Store cost incurred
                self.trades += 1

                print(f"[INFO] Sold {sell_shares} shares at {price:.2f}, total revenue: {revenue:.2f}")
            else:
                print("[WARNING] Not enough shares to execute sell order.")

        else:
            print("[INFO] No trade executed (Hold or magnitude too low).")

    def _make_plot(self):
        """
        Create and save a plot of the account value over time.

        This function helps visualize the portfolio's performance over an episode.
        The generated plot is saved to the 'results' directory.
        """

        print(f"[INFO] Generating account value plot for Episode {self.episode}...")

        plt.figure(figsize=(10, 6))
        plt.plot(self.asset_memory, "r", label="Account Value")
        plt.title(f"Account Value - Episode {self.episode}")
        plt.xlabel("Time Steps")
        plt.ylabel("Account Value")
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plot_filename = f"results/account_value_trade_{self.episode}.png"
        plt.savefig(plot_filename)
        plt.close()

        print(f"[INFO] Plot saved: {plot_filename}")

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment for a new episode.

        This function resets:
        - The trading day to the start of the dataset.
        - The initial state with updated stock data.
        - Asset tracking variables.
        - Tracking for trades, costs, and rewards.

        Returns:
            tuple: (state, info)
        """

        print("\n[INFO] Resetting the environment for a new episode...")

        # Set random seed if provided
        super().reset(seed=seed)

        # Reset to the first trading day
        self.day = 0
        self.data = self.df.loc[self.day, :]
        print(f"[INFO] Reset to Day {self.day}: {self.data.name}")

        # Initialize the state
        self.state = self._initiate_state()
        print("[INFO] Initialized state:", self.state)

        # Initialize asset memory with starting cash + stock holdings value
        self.asset_memory = [
            self.initial_amount + (self.state[1] * self.state[self.stock_dim + 1])
        ]
        print(f"[INFO] Initial portfolio value: {self.asset_memory[0]:.2f}")

        # Reset tracking variables (reward, cost, trades, etc.)
        self.reset_tracking_variables()

        # Increment episode count
        self.episode += 1
        print(f"[INFO] Starting Episode {self.episode}")

        return np.array(self.state, dtype=np.float32), {}

    def _update_state(self):
        """
        Update the state representation with the current day's stock data.

        The state includes:
        - Cash balance.
        - Current stock price.
        - Previous stock prices.
        - Technical indicators.

        Returns:
            list: The updated state representation.
        """

        print(f"[INFO] Updating state for Day {self.day}...")

        state = (
                [self.state[0]]  # Cash balance
                + [self.data.close]  # Current stock price
                + list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])  # Previous prices
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])  # Technical indicators
        )

        print("[INFO] Updated state:", state)
        return state

    def _get_date(self):
        """
        Retrieve the current trading date.

        Returns:
            str: The date corresponding to the current trading day.
        """

        current_date = self.df.index[self.day]
        print(f"[INFO] Retrieved current date: {current_date}")
        return current_date

    def render(self, mode="human", close=False):
        """
        Render the current state of the environment.

        Currently, this is a minimal implementation that just returns the state.
        If a visualization is required, this method can be extended.

        Args:
            mode (str): The rendering mode (default: "human").
            close (bool): Whether to close the rendering (default: False).

        Returns:
            list: The current environment state.
        """
        print(f"[INFO] Rendering current state: {self.state}")
        return self.state

    def seed(self, seed=None):
        """
        Set a random seed for reproducibility.

        This ensures that the environment behaves consistently across different runs.

        Args:
            seed (int, optional): Seed value for random number generator.

        Returns:
            list: The seed used.
        """
        print(f"[INFO] Setting environment seed: {seed}")
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """
        Create a Stable Baselines-compatible environment.

        This wraps the current environment in a DummyVecEnv, which is required
        for using RL algorithms from the Stable Baselines3 library.

        Returns:
            tuple: (DummyVecEnv environment, initial observation)
        """
        print("[INFO] Creating Stable Baselines-compatible environment...")
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        print("[INFO] Environment initialized for Stable Baselines.")
        return e, obs

    def save_state_memory(self):
        """
        Save the recorded state memory as a Pandas DataFrame.

        This is useful for analyzing how the state evolved over time.

        Returns:
            pd.DataFrame: A DataFrame containing recorded states with timestamps.
        """
        print("[INFO] Saving state memory...")
        date_list = self.date_memory[:-1]
        state_list = self.state_memory
        df = pd.DataFrame({"date": date_list, "states": state_list})
        print(f"[INFO] State memory saved. Total records: {len(df)}")
        return df

    def save_asset_memory(self):
        """
        Save the account value over time as a Pandas DataFrame.

        This helps in tracking portfolio performance across different time steps.

        Returns:
            pd.DataFrame: A DataFrame with account values over time.
        """
        print("[INFO] Saving asset memory (account value over time)...")
        df = pd.DataFrame({
            "date": self.date_memory,
            "account_value": self.asset_memory
        })
        print(f"[INFO] Asset memory saved. Total records: {len(df)}")
        return df

    def save_action_memory(self):
        """
        Save the history of actions taken as a Pandas DataFrame.

        This is useful for analyzing the agent's behavior over time.

        Returns:
            pd.DataFrame: A DataFrame containing recorded actions with timestamps.
        """
        print("[INFO] Saving action memory...")
        date_list = self.date_memory[:-1]
        df = pd.DataFrame({
            "date": date_list,
            "actions": self.actions_memory
        })
        print(f"[INFO] Action memory saved. Total records: {len(df)}")
        return df
