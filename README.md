# Kaggle Spaceship Titanic
Target prediction is 'Transported'

## Preprocessing 
* Filled in missing values with 'Missing' or 0
* Dropped Name column due to high cardinality and low value
* Split up Cabin data (deck/floor/side)
* Replaces missing ages with calculated mean

Original Data Table:
```
PassengerId HomePlanet CryoSleep  Cabin  Destination   Age  ... FoodCourt  ShoppingMall     Spa  VRDeck               Name  Transported
0     0001_01     Europa     False  B/0/P  TRAPPIST-1e  39.0  ...       0.0           0.0     0.0     0.0    Maham Ofracculy        False
1     0002_01      Earth     False  F/0/S  TRAPPIST-1e  24.0  ...       9.0          25.0   549.0    44.0       Juanna Vines         True
2     0003_01     Europa     False  A/0/S  TRAPPIST-1e  58.0  ...    3576.0           0.0  6715.0    49.0      Altark Susent        False
3     0003_02     Europa     False  A/0/S  TRAPPIST-1e  33.0  ...    1283.0         371.0  3329.0   193.0       Solam Susent        False
4     0004_01      Earth     False  F/1/S  TRAPPIST-1e  16.0  ...      70.0         151.0   565.0     2.0  Willy Santantines         True
```

Data Table after preprocessing:
```
PassengerId HomePlanet CryoSleep  Destination   Age    VIP  RoomService  FoodCourt  ShoppingMall     Spa  VRDeck  Transported Deck Side
0     0001_01     Europa     False  TRAPPIST-1e  39.0  False          0.0        0.0           0.0     0.0     0.0        False    B    P
1     0002_01      Earth     False  TRAPPIST-1e  24.0  False        109.0        9.0          25.0   549.0    44.0         True    F    S
2     0003_01     Europa     False  TRAPPIST-1e  58.0   True         43.0     3576.0           0.0  6715.0    49.0        False    A    S
3     0003_02     Europa     False  TRAPPIST-1e  33.0  False          0.0     1283.0         371.0  3329.0   193.0        False    A    S
4     0004_01      Earth     False  TRAPPIST-1e  16.0  False        303.0       70.0         151.0   565.0     2.0         True    F    S
```

## Modeling
Models uses Random Forest Classifier and Gradient Boosting Classifier

## Model Peformance
```
Metrics for rf, Accuracy - 0.7963957055214724, Recall - 0.7723765432098766, Precision - 0.809215844785772
Metrics for gb, Accuracy - 0.8071319018404908, Recall - 0.8472222222222222, Precision - 0.782608695652174
```