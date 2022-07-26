
import pandas as pd


def scraper_create_labels_hermite(data):

    def make_group(row):

        row = row.price_dkk
        if row < Q1:
            return 1
        if row < Q2:
            return 2
        if row < Q3:
            return 3
        else:
            return 4

    data.loc[:, "y"] = data.apply(lambda x: make_group(x), axis=1)
    labels = data.y

    del data["price_dkk"]
    del data["y"]

    return data, labels


def train_test_data(data, test_p=0.2):

    N = data.shape[0]
    split = int(N * (1-test_p))

    if isinstance(data, pd.Series):
        train_data = data.iloc[:split]
        test_data = data.iloc[split:]
    else:
        train_data = data.iloc[:split, :]
        test_data = data.iloc[split:, :]

    return train_data, test_data


data = pd.read_excel("data/ANN_data.xlsx", index_col=False)
data = data.sample(frac=1)   # Shuffling data

# Remove outliers using IQR - boundaries found in R
data = data[data.price_dkk > 3085.5]
data = data[data.price_dkk < 8018]

# From R script (Hermite Density Quartiles)
Q1 = 5099.55
Q2 = 5597.65
Q3 = 6059.80

data, labels = scraper_create_labels_hermite(data)

# Train is 80% of data, Test 20%
delta = 0.2
X_train, X_test = train_test_data(data, delta)
y_train, y_test = train_test_data(labels, delta)

X_train.to_csv("data/COPY_train_auctions.csv", index=False)
y_train.to_csv("data/COPY_train_labels.csv", index=False)

X_test.to_csv("data/COPY_test_auctions.csv", index=False)
y_test.to_csv("data/COPY_test_labels.csv", index=False)