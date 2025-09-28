import matplotlib.pyplot as plt
from data_pipeline import load_data, add_indicators
from strategy import sma_crossover
from ml_model import train_model

# 1. Load & prepare data
data = load_data("USO")
data = add_indicators(data)

# 2. Apply SMA crossover strategy
data = sma_crossover(data)

# 3. Backtest portfolio
initial_capital = 10000
data["Strategy_Returns"] = data["Signal"].shift(1).astype(float) * data["Close"].pct_change()
data["Portfolio"] = initial_capital * (1 + data["Strategy_Returns"]).cumprod()
data["Buy_Hold"] = initial_capital * (1 + data["Close"].pct_change()).cumprod()

plt.figure(figsize=(12,6))
plt.plot(data["Portfolio"], label="Strategy")
plt.plot(data["Buy_Hold"], label="Buy & Hold")
plt.legend()
plt.show()

# 4. Train ML model
clf, acc = train_model(data)
print("Model Accuracy:", acc)
sharpe = (data["Strategy_Returns"].mean() / data["Strategy_Returns"].std()) * (252**0.5)
max_drawdown = (data["Portfolio"] / data["Portfolio"].cummax() - 1).min()
print("Sharpe Ratio:", sharpe)
print("Max Drawdown:", max_drawdown)
