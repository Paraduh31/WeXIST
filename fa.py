import pandas as pd   
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

csv_path = "nifty_500.csv"
nse_stocks_df = pd.read_csv(csv_path)

nse_stocks_df.columns = nse_stocks_df.columns.str.strip()

print(nse_stocks_df.columns)

stock_column = "SYMBOL"

nse_stocks = [symbol.strip() + ".NS" for symbol in nse_stocks_df[stock_column].tolist()]

stock_data = []
for stock in nse_stocks:
    try:
        ticker = yf.Ticker(stock)
        info = ticker.info
        
        if info and 'marketCap' in info:
            stock_data.append({
                "Stock Name": stock,
                "Market Cap": info.get("marketCap", np.nan),
                "Revenue Growth": info.get("revenueGrowth", np.nan),
                "EPS Growth": info.get("earningsGrowth", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "Debt-to-Equity": info.get("debtToEquity", np.nan),
                "P/E Ratio": info.get("trailingPE", np.nan),
                "P/B Ratio": info.get("priceToBook", np.nan),
                "Dividend Yield": info.get("dividendYield", np.nan),
                "Future Returns": np.random.uniform(0, 1)
            })
    except Exception as e:
        print(f"⚠️ Error fetching {stock}: {e}")

df = pd.DataFrame(stock_data).dropna()

features = ["Market Cap", "Revenue Growth", "EPS Growth", "ROE", "Debt-to-Equity", 
            "P/E Ratio", "P/B Ratio", "Dividend Yield"]
target = "Future Returns"

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror", 
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    scale_pos_weight=1,
    booster="gbtree",
    tree_method="auto",
    reg_alpha=0,
    reg_lambda=1
)
xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(df[features])

df["Predicted Returns"] = xgb_predictions
top_stocks = df.sort_values(by="Predicted Returns", ascending=False).head(10)

print("\n🔝 Top 10 Stocks Based on XGBoost Model:")
print(top_stocks[["Stock Name", "Predicted Returns"]])