import pandas as pd

#Loading the data
matches = pd.read_csv("matches.csv", index_col=0)

matches.head()

matches.shape

# 2 seasons * 20 squads * 38 matches
#investigating the missing data
2 * 20 * 38

# Missing Liverpool 2021-2022
#According to above output 1520 matches should be there but there are less 1389 matches
matches["team"].value_counts()

matches[matches["team"] == "Liverpool"].sort_values("date")

matches["round"].value_counts()

#cleaning the data for machine learning
matches.dtypes

#creating predictors for machine learning
del matches["comp"]

del matches["notes"]

matches["date"] = pd.to_datetime(matches["date"])

matches["target"] = (matches["result"] == "W").astype("int")

matches

matches["venue_code"] = matches["venue"].astype("category").cat.codes

matches["opp_code"] = matches["opponent"].astype("category").cat.codes

matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

matches["day_code"] = matches["date"].dt.dayofweek

matches

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
#n_estimators=no. of individual decision trees we want to train
#higher this is = longer time for algo to run = more accurate it will be

train = matches[matches["date"] < '2022-01-01']

test = matches[matches["date"] > '2022-01-01']

predictors = ["venue_code", "opp_code", "hour", "day_code"]

rf.fit(train[predictors], train["target"])

preds = rf.predict(test[predictors])

#if we predicted win what percentage of time we are correct
from sklearn.metrics import accuracy_score

error = accuracy_score(test["target"], preds)
#acc,pred

error

combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))
#combining the actual values and predicted values

pd.crosstab(index=combined["actual"], columns=combined["predicted"])

#we need to revise our model
from sklearn.metrics import precision_score

precision_score(test["target"], preds)

grouped_matches = matches.groupby("team")

group = grouped_matches.get_group("Manchester City").sort_values("date")

#calculating rolling averages
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()  #closed means keeping the current week out of average
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) #drop all the rows with missing values
    return group

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))

matches_rolling

matches_rolling = matches_rolling.droplevel('team')

matches_rolling

matches_rolling.index = range(matches_rolling.shape[0])

#retaining our ml pred
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error

combined, error = make_predictions(matches_rolling, predictors + new_cols)

error

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

combined.head(10)

#mapping the names which are different
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Tottenham Hotspur": "Tottenham", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"}
mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping)

#we will see if the prediction are matching up on both sides
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])

merged

merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()











