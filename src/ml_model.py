from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(data):
    features = ["SMA_20", "SMA_50", "Return"]
    data = data.dropna()
    X = data[features]
    y = (data["Return"].shift(-1) > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return clf, accuracy
