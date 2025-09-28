def sma_crossover(data):
    data["Signal"] = 0
    data.loc[data.index[20]:, "Signal"] = (data["SMA_20"].iloc[20:] > data["SMA_50"].iloc[20:]).astype(int)
    data["Position"] = data["Signal"].diff()
    return data
