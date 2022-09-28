import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
# from sklearn.multioutput import MultiOutputRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline

def convert_object_types(df):
    teams = {
        'Blue': 100, 
        'Red': 200,
        'No team': 300
        }
    t = {
        'CHAMPION_KILL': 0, 
        'CHAMPION_SPECIAL_KILL': 1, 
        'TURRET_PLATE_DESTROYED': 2, 
        'BUILDING_KILL': 3, 
        'ELITE_MONSTER_KILL': 4
        }
    killT = {
        "KILL_FIRST_BLOOD": 0, 
        "1_KILL": 1, 
        "2_KILL": 2, 
        "3_KILL": 3, 
        "4_KILL": 4, 
        "5_KILL": 5, 
        "6_KILL": 6, 
        "7_KILL": 7, 
        "8_KILL": 8, 
        "9_KILL": 9, 
        "10_KILL": 10, 
        "TURRET_PLATE": 11, 
        "DRAGON": 12,
        "RIFTHERALD": 13,
        "BARON_NASHOR": 14,
        "KILL_MULTI": 15,
        "TOWER_BUILDING": 16,
        "INHIBITOR_BUILDING": 17,
        "KILL_ACE": 18
        }

    # one_hot = pd.get_dummies(df['type'])
    # df = df.drop('type', axis = 1)
    # df = df.join(one_hot)
    df.type = [t[item] for item in df.type]
    df.killerTeam = [teams[item] for item in df.killerTeam]
    df.killType = [killT[item] for item in df.killType]
    
    df['quadrant_x'] = np.select([(df.x > 0) & (df.x <= 2000),
                                (df.x > 2000) & (df.x <= 4000),
                                (df.x > 4000) & (df.x <= 6000),
                                (df.x > 6000) & (df.x <= 8000),
                                (df.x > 8000) & (df.x <= 10000),
                                (df.x > 10000) & (df.x <= 12000),
                                (df.x > 12000) & (df.x <= 14000)],
                                choicelist = [1, 2, 3, 4, 5, 6, 7])

    df['quadrant_y'] = np.select([(df.y > 0) & (df.y <= 2000),
                                (df.y > 2000) & (df.y <= 4000),
                                (df.y > 4000) & (df.y <= 6000),
                                (df.y > 6000) & (df.y <= 8000),
                                (df.y > 8000) & (df.y <= 10000),
                                (df.y > 10000) & (df.y <= 12000),
                                (df.y > 12000) & (df.y <= 14000)],
                                choicelist = [1, 2, 3, 4, 5, 6, 7])

    df['quadrant'] = np.select([(df.quadrant_x == 1) & (df.quadrant_y == 1),
                                (df.quadrant_x == 2) & (df.quadrant_y == 1),
                                (df.quadrant_x == 3) & (df.quadrant_y == 1),
                                (df.quadrant_x == 4) & (df.quadrant_y == 1),
                                (df.quadrant_x == 5) & (df.quadrant_y == 1),
                                (df.quadrant_x == 6) & (df.quadrant_y == 1),
                                (df.quadrant_x == 7) & (df.quadrant_y == 1),
                                (df.quadrant_x == 1) & (df.quadrant_y == 2),
                                (df.quadrant_x == 2) & (df.quadrant_y == 2),
                                (df.quadrant_x == 3) & (df.quadrant_y == 2),
                                (df.quadrant_x == 4) & (df.quadrant_y == 2),
                                (df.quadrant_x == 5) & (df.quadrant_y == 2),
                                (df.quadrant_x == 6) & (df.quadrant_y == 2),
                                (df.quadrant_x == 7) & (df.quadrant_y == 2),
                                (df.quadrant_x == 1) & (df.quadrant_y == 3),
                                (df.quadrant_x == 2) & (df.quadrant_y == 3),
                                (df.quadrant_x == 3) & (df.quadrant_y == 3),
                                (df.quadrant_x == 4) & (df.quadrant_y == 3),
                                (df.quadrant_x == 5) & (df.quadrant_y == 3),
                                (df.quadrant_x == 6) & (df.quadrant_y == 3),
                                (df.quadrant_x == 7) & (df.quadrant_y == 3),
                                (df.quadrant_x == 1) & (df.quadrant_y == 4),
                                (df.quadrant_x == 2) & (df.quadrant_y == 4),
                                (df.quadrant_x == 3) & (df.quadrant_y == 4),
                                (df.quadrant_x == 4) & (df.quadrant_y == 4),
                                (df.quadrant_x == 5) & (df.quadrant_y == 4),
                                (df.quadrant_x == 6) & (df.quadrant_y == 4),
                                (df.quadrant_x == 7) & (df.quadrant_y == 4),
                                (df.quadrant_x == 1) & (df.quadrant_y == 5),
                                (df.quadrant_x == 2) & (df.quadrant_y == 5),
                                (df.quadrant_x == 3) & (df.quadrant_y == 5),
                                (df.quadrant_x == 4) & (df.quadrant_y == 5),
                                (df.quadrant_x == 5) & (df.quadrant_y == 5),
                                (df.quadrant_x == 6) & (df.quadrant_y == 5),
                                (df.quadrant_x == 7) & (df.quadrant_y == 5),
                                (df.quadrant_x == 1) & (df.quadrant_y == 6),
                                (df.quadrant_x == 2) & (df.quadrant_y == 6),
                                (df.quadrant_x == 3) & (df.quadrant_y == 6),
                                (df.quadrant_x == 4) & (df.quadrant_y == 6),
                                (df.quadrant_x == 5) & (df.quadrant_y == 6),
                                (df.quadrant_x == 6) & (df.quadrant_y == 6),
                                (df.quadrant_x == 7) & (df.quadrant_y == 6),
                                (df.quadrant_x == 1) & (df.quadrant_y == 7),
                                (df.quadrant_x == 2) & (df.quadrant_y == 7),
                                (df.quadrant_x == 3) & (df.quadrant_y == 7),
                                (df.quadrant_x == 4) & (df.quadrant_y == 7),
                                (df.quadrant_x == 5) & (df.quadrant_y == 7),
                                (df.quadrant_x == 6) & (df.quadrant_y == 7),
                                (df.quadrant_x == 7) & (df.quadrant_y == 7)],
                                choicelist = range(1, 50))

    df.drop('x', axis = 'columns', inplace = True)
    df.drop('y', axis = 'columns', inplace = True)
    df.drop('quadrant_x', axis = 'columns', inplace = True)
    df.drop('quadrant_y', axis = 'columns', inplace = True)

    return df

def linear_regression(X, y, scores):
    lr = (LinearRegression())
    print("Linear regression:")
    cross_validation(lr, X, y, scores)
    print()
    lr.fit(X,y)

    return lr

def descision_tree(X, y, scores):
    dt = (DecisionTreeRegressor(random_state = 42))
    print("Descision tree:")
    cross_validation(dt, X, y, scores)
    print()
    dt.fit(X,y)

    return dt

def random_forest(X, y, scores):
    rf = (RandomForestRegressor(max_depth = 2, random_state = 42))
    print("Random forest:")
    cross_validation(rf, X, y, scores)
    print()
    rf.fit(X,y)

    return rf

def support_vector_machine(X, y, scores):
    svm = (SVR(gamma = 'auto', kernel='rbf'))
    print("Support Vector Machine:")
    cross_validation(svm, X, y, scores)
    print()
    svm.fit(X, y)

    return svm

def scaling(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    scaled_Data = scaler.transform(data)

    return scaled_Data

def score_metrics(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, y_pred, 'o')
    plt.plot([-10, 60], [-10, 60], 'k--')
    plt.axis([-10, 60, -10, 60])
    plt.xlabel('ground truth')
    plt.ylabel('predicted')
    plt.title(title)

    scorestr = r'R$^2$ = %.3f' % r2
    errstr = 'MSE = %.3f' % mse
    img_path = "Images\\{}.png".format(title)
    plt.text(-5, 50, scorestr, fontsize=12)
    plt.text(-5, 45, errstr, fontsize=12)
    plt.show()
    # plt.savefig(img_path, bbox_inches = 'tight')
    # plt.close()

    return r2, mae, mse

def lstm(X, y):
    X, y = np.array(X), np.array(y)
    X= np.reshape(X, (X.shape[0],X.shape[1],1))
    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1],1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
    regressor.fit(X,y,epochs=50,batch_size=32)

    return regressor

def cross_validation(model, X, y, scores):
    cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
    for i in scores:
        score = cross_val_score(model, X, y, scoring = i, cv = cv, n_jobs = -1)
        print("Mean training %s: %.3f (%.3f)" % (i, score.mean(), score.std()))

if __name__ == "__main__":
    train_df = pd.read_csv("CSV\\timeline_data_large.csv")
    test_df = pd.read_csv("CSV\\timeline_data_mini.csv")
    # train_df = pd.concat([train_df.loc[(train_df['killerId'] == 2)], train_df.loc[(train_df['killerId'] == 7)]])
    # test_df = pd.concat([test_df.loc[(test_df['killerId'] == 2)], test_df.loc[(test_df['killerId'] == 7)]])

    # print(train_df.head())
    # print(train_df.shape)
    # print(train_df.info())

    train_df = convert_object_types(train_df)
    test_df = convert_object_types(test_df)

    # print(train_df.head())
    # quit()
    # print(train_df.info())

    # sns.pairplot(train_df)
    # plt.show()

    # y_train = pd.concat([train_df.pop(i) for i in ['quadrant_x', 'quadrant_y']], axis = 1)
    # y_train = pd.concat([train_df.pop(i) for i in ['type', 'killType']], axis = 1)
    y_train = train_df.pop('quadrant')
    X_train = train_df
    # y_test = pd.concat([test_df.pop(i) for i in ['quadrant_x', 'quadrant_y']], axis = 1)
    # y_test = pd.concat([test_df.pop(i) for i in ['type', 'killType']], axis = 1)
    y_test = test_df.pop('quadrant')
    X_test = test_df

    # print(X_train.head())

    X_train = scaling(X_train)
    X_test = scaling(X_test)

    scores = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]

    lr_trained = linear_regression(X_train, y_train, scores)
    dt_trained = descision_tree(X_train, y_train, scores)
    rf_trained = random_forest(X_train, y_train, scores)
    svm_trained = support_vector_machine(X_train, y_train, scores)
    lstm_trained = lstm(X_train, y_train)

    models = [lr_trained, dt_trained, rf_trained, svm_trained]
    model_names = ["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine"]

    for i in range(len(models)):
        r2, mae, mse = score_metrics(models[i], X_test, y_test, model_names[i])
        print("R2 score for model", model_names[i] ,":", r2)
        print("Mean absolute error for model", model_names[i] ,":", mae)
        print("Mean squared error for model", model_names[i] ,":", mse)
        print()      

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    r2, mae, mse = score_metrics(lstm_trained, X_test, y_test, "LSTM")

    print("R2 score for model LSTM:", r2)
    print("Mean absolute error for model LSTM:", mae)
    print("Mean squared error for model LSTM:", mse)