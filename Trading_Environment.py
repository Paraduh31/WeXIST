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
            df: pd.DataFrame,stock_dim: int, hmax: int, initial_amount: float, num_stock_shares: list[int], buy_cost_pct: list[float], sell_cost_pct: list[float], reward_scaling: float, state_space: int,tech_indicator_list: list[str],make_plots: bool = False,print_verbosity=10,day=0,initial=True,previous_state=None,model_name="",mode="", iteration="" ):
        super(StockTradingEnv, self).__init__()
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

        self.action_space = spaces.Box(
            low=np.array([-1, 0]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )

        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.state = self._initiate_state()

        self.reset_tracking_variables()

    def reset_tracking_variables(self):
        self.reward = 0
        self.cost = 0
        self.trades = 0

        self.episode  += 1

        self.asset_memory = [self.initial_amount]

        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]

    def _initiate_state(self):
        if self.initial:
            state = [
                self.initial_amount,
                *self.num_stock_shares,
                self.data['close'],
                *[self.data[tech] for tech in self.tech_indicator_list],
            ]
        else:
            state = [
                self.previous_state[0],
                *self.previous_state[1:self.stock_dim + 1],
                self.data['close'],
                *[self.data[tech] for tech in self.tech_indicator_list],
            ]
        return state

    def step(self, action):
        self._execute_action(action)

        self.day += 1

        if self.day >= len(self.df)-1 :
            self.terminal = True
        else:
            self.terminal = False

        if not self.terminal:
            self.data = self.df.loc[self.day, :]
            self.state = self._update_state()

        current_portfolio_value = (
                self.state[0] +
                np.sum(np.array(self.state[1:self.stock_dim + 1]) *
                       np.array(self.state[self.stock_dim + 1:self.stock_dim * 2 + 1]))
        )

        prev_portfolio_value = self.asset_memory[-1] if len(self.asset_memory) > 1 else self.initial_amount
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward *= self.reward_scaling

        self.asset_memory.append(current_portfolio_value)
        self.rewards_memory.append(reward)
        self.date_memory.append(self._get_date())
        self.state_memory.append(self.state)
        self.actions_memory.append(action)

        if self.terminal and self.make_plots:
            self._make_plot()

        self.cost = 0
        self.trades = 0

        return (
            np.array(self.state, dtype=np.float32),
            reward,
            self.terminal,
            False,
            {"date": self._get_date(), "portfolio_value": current_portfolio_value}
        )

    def _execute_action(self, action):
        direction, magnitude = action[0], action[1]

        cash = self.state[0]
        shares = int(self.state[1])
        price = self.state[self.stock_dim + 1]
        buy_cost = self.buy_cost_pct[0]
        sell_cost = self.sell_cost_pct[0]

        if direction > 0:
            investable_amount = cash * magnitude
            max_shares = int(
                np.floor(investable_amount / (price * (1 + buy_cost))))
            buy_shares = min(max_shares, self.hmax)
            total_cost = buy_shares * price * (1 + buy_cost)

            if total_cost <= cash:
                self.state[0] -= total_cost
                self.state[1] = int(self.state[1] + buy_shares)
                self.cost += total_cost - (buy_shares * price)
                self.trades += 1 if buy_shares > 0 else 0

        elif direction < 0:
            sell_shares = int(np.floor(shares * magnitude))
            sell_shares = min(sell_shares, self.hmax)
            revenue = sell_shares * price * (1 - sell_cost)

            if sell_shares > 0:
                self.state[0] += revenue
                self.state[1] = int(self.state[1] - sell_shares)
                self.cost += sell_shares * price * sell_cost
                self.trades += 1

    def _make_plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.asset_memory, "r", label="Account Value")
        plt.title(f"Account Value - Episode {self.episode}")
        plt.xlabel("Time Steps")
        plt.ylabel("Account Value")
        plt.legend()
        plt.tight_layout()

        plot_filename = f"results/account_value_trade_{self.episode}.png"
        plt.savefig(plot_filename)
        plt.close()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.day = 0
        self.data = self.df.loc[self.day, :]

        self.state = self._initiate_state()

        self.asset_memory = [
            self.initial_amount + (self.state[1] * self.state[self.stock_dim + 1])
        ]

        self.reset_tracking_variables()

        self.episode += 1

        return np.array(self.state, dtype=np.float32), {}

    def _update_state(self):
        state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
        )

        return state

    def _get_date(self):
        current_date = self.df.index[self.day]
        return current_date

    def render(self, mode="human", close=False):
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def save_state_memory(self):
        date_list = self.date_memory[:-1]
        state_list = self.state_memory
        df = pd.DataFrame({"date": date_list, "states": state_list})
        return df

    def save_asset_memory(self):
        df = pd.DataFrame({
            "date": self.date_memory,
            "account_value": self.asset_memory
        })
        return df

    def save_action_memory(self):
        date_list = self.date_memory[:-1]
        df = pd.DataFrame({
            "date": date_list,
            "actions": self.actions_memory
        })
        return df
