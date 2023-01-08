# File contains various experiments with different neural network configurations for the NFL data
# as well as use of sklearn methods for comparison
import pandas as pd
import GameCSVBuilder
import numpy as np
import NFL_NN_2
import torch
import torch.nn as nn
import NFL_Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


# experiment 2 runs a 3-layer multilayer perceptron, experimenting with the number of neurons
# in the middle layer
def experiment_2():
    training_percent = 0.85
    batch_size = 128
    file_name = "../Outputs/labeled_data.csv"
    loaders = NFL_Dataset.get_train_test_loaders(file_name, training_percent, batch_size)
    input_size = len(loaders[0].dataset[0][0])
    num_epochs = 30
    loss_function = nn.SmoothL1Loss(reduction="sum")
    csv_output = "hid_layer_neurons,best_test_loss,best_pred_accuracy\n"
    for num_hid_layers in range(2, input_size+1):
        nfl_mlp = NFL_NN_2.NFL_MLP(loaders, 3, [input_size, num_hid_layers, 1], nn.ReLU, False)
        optimizer = torch.optim.Adam(nfl_mlp.parameters(), lr=0.0008)
        numpy_results = nfl_mlp.train_nn(loss_function, optimizer, num_epochs)
        best_test_loss = np.min(numpy_results[:, 1])
        best_prediction_rate = np.max(numpy_results[:, 2])
        csv_output += f"{num_hid_layers},{best_test_loss},{best_prediction_rate}\n"
    test_results = open("../ExperimentResults/experiment_2.csv", "w")
    test_results.write(csv_output)
    test_results.close()


# Experiment 3 concerns a 4-layer NN and loops through various combinations of number of neurons
# in middle 2 layers
def experiment_3():
    training_percent = 0.85
    batch_size = 128
    file_name = "../Outputs/labeled_data.csv"
    loaders = NFL_Dataset.get_train_test_loaders(file_name, training_percent, batch_size)
    input_size = len(loaders[0].dataset[0][0])
    num_epochs = 20
    loss_function = nn.SmoothL1Loss(reduction="sum")
    csv_output = "layer_2_neurons,layer_3_neurons,best_test_loss,best_pred_accuracy\n"
    for layer_2_neurons in range(2, input_size, 2):
        for layer_3_neurons in range(2, input_size, 2):
            nfl_mlp = NFL_NN_2.NFL_MLP(loaders, 4, [input_size, layer_2_neurons, layer_3_neurons, 1], nn.ReLU, False)
            optimizer = torch.optim.Adam(nfl_mlp.parameters(), lr=0.0008)
            numpy_results = nfl_mlp.train_nn(loss_function, optimizer, num_epochs)
            best_test_loss = np.min(numpy_results[:, 1])
            best_prediction_rate = np.max(numpy_results[:, 2])
            csv_output += f"{layer_2_neurons},{layer_3_neurons},{best_test_loss},,{best_prediction_rate}\n"
    test_results = open("../ExperimentResults/experiment_3.csv", "w")
    test_results.write(csv_output)
    test_results.close()


# same as experiment 2, but with classification instead of regression
def experiment_4():
    training_percent = 0.85
    batch_size = 128
    file_name = "../Outputs/labeled_data_binary.csv"
    loaders = NFL_Dataset.get_train_test_loaders(file_name, training_percent, batch_size)
    input_size = len(loaders[0].dataset[0][0])
    num_epochs = 20
    loss_function = nn.BCELoss(reduction='sum')
    csv_output = "hid_layer_neurons,best_test_loss,best_pred_accuracy\n"
    for num_hid_layers in range(2, input_size + 1):
        nfl_mlp = NFL_NN_2.NFL_MLP(loaders, 3, [input_size, num_hid_layers, 1], nn.ReLU, True)
        optimizer = torch.optim.Adam(nfl_mlp.parameters(), lr=0.0008)
        numpy_results = nfl_mlp.train_nn(loss_function, optimizer, num_epochs)
        best_test_loss = np.min(numpy_results[:, 1])
        best_prediction_rate = np.max(numpy_results[:, 2])
        csv_output += f"{num_hid_layers},{best_test_loss},{best_prediction_rate}\n"
    test_results = open("../ExperimentResults/experiment_4.csv", "w")
    test_results.write(csv_output)
    test_results.close()


# experiment 5 uses a smaller dataset, determined by less influential factors as known from experiment 6 (random
# forest)
def experiment_5():
    training_percent = 0.85
    batch_size = 128
    GameCSVBuilder.build_smaller_dim_data(True)
    file_name = "../Outputs/labeled_data_small_binary.csv"
    loaders = NFL_Dataset.get_train_test_loaders(file_name, training_percent, batch_size)
    input_size = len(loaders[0].dataset[0][0])
    num_epochs = 30
    loss_function = nn.BCELoss(reduction="sum")
    csv_output = "hid_layer_neurons,best_test_loss,best_pred_accuracy\n"
    for num_hid_layers in range(2, input_size + 1):
        nfl_mlp = NFL_NN_2.NFL_MLP(loaders, 3, [input_size, num_hid_layers, 1], nn.ReLU, True)
        optimizer = torch.optim.Adam(nfl_mlp.parameters(), lr=0.0008)
        numpy_results = nfl_mlp.train_nn(loss_function, optimizer, num_epochs)
        best_test_loss = np.min(numpy_results[:, 1])
        best_prediction_rate = np.max(numpy_results[:, 2])
        csv_output += f"{num_hid_layers},{best_test_loss},{best_prediction_rate}\n"
    test_results = open("../ExperimentResults/experiment_5.csv", "w")
    test_results.write(csv_output)
    test_results.close()


# Experiment 6 uses random forest from sklearn and then xgboost on the same dataset
def experiment_6():
    per_season_df = pd.read_csv("../Outputs/per_team_per_season_stats.csv")
    up_to_game_df = pd.read_csv("../Outputs/game_up_to_now_stats.csv")
    GameCSVBuilder.build_per_game_per_team_df(up_to_game_df, per_season_df, True)
    GameCSVBuilder.build_df_for_sklearn("../Outputs/labeled_df", True)
    labeled_data_df = pd.read_csv("../Outputs/labeled_df_binary.csv")\
        .drop(["gold_label_them"], axis=1)
    labels = labeled_data_df["gold_label_us"]
    features = labeled_data_df.loc[:, "season_games_to_now_us":"tackles_for_loss_prior_s_them"]
    train_data, test_data, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.15, random_state=0)
    rf_class = RandomForestClassifier(n_estimators=1000, random_state=0)
    rf_class.fit(train_data, train_labels)
    accuracy = rf_class.score(test_data, test_labels)
    print(f"Accuracy of random forest: {accuracy}")
    feature_importance = pd.Series(rf_class.feature_importances_, index=features.columns).sort_values(ascending=False)
    feature_importance.to_csv("../Outputs/feature_importance_rf.csv")
    xgb_classifier = xgb.XGBClassifier(n_estimators=1000)
    xgb_classifier.fit(train_data, train_labels)
    accuracy = xgb_classifier.score(test_data, test_labels)
    print(f"Accuracy of xgb classifier is {accuracy}")
    xgb_rf = xgb.XGBRFClassifier(n_estimators=1000)
    xgb_rf.fit(train_data, train_labels)
    accuracy = xgb_rf.score(test_data, test_labels)
    print(f"Accuracy of xgb random forest is {accuracy}")


def sklearn_helper(classifier, split_data):
    train_data = split_data[0]
    test_data = split_data[1]
    train_labels = split_data[2]
    test_labels = split_data[3]
    classifier.fit(train_data, train_labels)
    accuracy = classifier.score(test_data, test_labels)
    print(f"Accuracy of {classifier}: {accuracy}")


# experiment_7 uses logistic regression, SVM, Gaussian Naive Bayes, and K-nearest neighbors
# from sklearn on same dataset
def experiment_7():
    labeled_data_df = pd.read_csv("../Outputs/labeled_data_df.csv") \
        .drop(["gold_label_them"], axis=1)
    labels = labeled_data_df["gold_label_us"]
    features = labeled_data_df.loc[:, "season_games_to_now_us":"tackles_for_loss_prior_s_them"]
    features = StandardScaler().fit_transform(features)
    split_data = train_test_split(features, labels, test_size=0.15, random_state=0)
    sklearn_helper(LogisticRegression(random_state=0, max_iter=100000), split_data)
    sklearn_helper(SVC(random_state=0, kernel='rbf'), split_data)
    sklearn_helper(GaussianNB(), split_data)
    sklearn_helper(KNeighborsClassifier(n_neighbors=20), split_data)
    sklearn_helper(GradientBoostingClassifier(), split_data)


# builds data for experiments 8, 9
def build_weighted_representations(is_classification: bool, alpha: float):
    games_df = pd.read_csv("../Outputs/per_game_stats.csv")
    per_season_df = pd.read_csv("../Outputs/per_team_per_season_stats.csv")
    weighted_games = GameCSVBuilder.build_weighted_game_vectors(games_df, per_season_df, alpha)
    GameCSVBuilder.build_labeled_data_numpy(weighted_games, "../Outputs/labeled_data_weighted", is_classification)
    GameCSVBuilder.build_weighted_df_for_sklearn(weighted_games, "../Outputs/labeled_df_weighted", is_classification)


# Calculates and then uses a weighted, recursive representation of the games in each season
# Then uses weighted representation to find optimal 3-layer NN for classification
# Same as experiment 4 but with new, weighted representation
def experiment_8():
    build_weighted_representations(True, 0.45)
    training_percent = 0.85
    batch_size = 128
    file_name = "../Outputs/labeled_data_weighted_binary.csv"
    loaders = NFL_Dataset.get_train_test_loaders(file_name, training_percent, batch_size)
    input_size = len(loaders[0].dataset[0][0])
    num_epochs = 20
    loss_function = nn.BCELoss(reduction='sum')
    csv_output = "hid_layer_neurons,best_test_loss,best_pred_accuracy\n"
    for num_hid_layers in range(2, input_size + 1):
        nfl_mlp = NFL_NN_2.NFL_MLP(loaders, 3, [input_size, num_hid_layers, 1], nn.ReLU, True)
        optimizer = torch.optim.Adam(nfl_mlp.parameters(), lr=0.0008)
        numpy_results = nfl_mlp.train_nn(loss_function, optimizer, num_epochs)
        best_test_loss = np.min(numpy_results[:, 1])
        best_prediction_rate = np.max(numpy_results[:, 2])
        csv_output += f"{num_hid_layers},{best_test_loss},{best_prediction_rate}\n"
    test_results = open("../ExperimentResults/experiment_8.csv", "w")
    test_results.write(csv_output)
    test_results.close()


def experiment_9():
    build_weighted_representations(True, 0.45)
    labeled_data_df = pd.read_csv("../Outputs/labeled_df_weighted_binary.csv")\
        .drop(["gold_label_them"], axis=1)
    labels = labeled_data_df["gold_label_us"]
    features = labeled_data_df.loc[:, "point_dif_us":"tackles_for_loss_them"]
    train_data, test_data, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.15, random_state=0)
    rf_class = RandomForestClassifier(n_estimators=1000, random_state=0)
    rf_class.fit(train_data, train_labels)
    accuracy = rf_class.score(test_data, test_labels)
    print(f"Accuracy of random forest: {accuracy}")
    feature_importance = pd.Series(rf_class.feature_importances_, index=features.columns).sort_values(ascending=False)
    feature_importance.to_csv("../Outputs/feature_importance.csv")


# Same as experiment 7 but calculates and uses weighted representation
def experiment_10():
    build_weighted_representations(True, 0.45)
    labeled_data_df = pd.read_csv("../Outputs/labeled_df_weighted_binary.csv") \
        .drop(["gold_label_them"], axis=1)
    labels = labeled_data_df["gold_label_us"]
    features = labeled_data_df.loc[:, "point_dif_us":"tackles_for_loss_them"]
    features = StandardScaler().fit_transform(features)
    split_data = train_test_split(features, labels, test_size=0.15, random_state=0)
    sklearn_helper(LogisticRegression(random_state=0, max_iter=100000), split_data)
    sklearn_helper(SVC(random_state=0, kernel='rbf'), split_data)
    sklearn_helper(GaussianNB(), split_data)
    sklearn_helper(KNeighborsClassifier(n_neighbors=20), split_data)
    sklearn_helper(GradientBoostingClassifier(), split_data)


# Experiment 11 does hyperparameter tuning for the rf model, the most successful one produced from above. Based on:
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
def experiment_11():
    labeled_data_df = pd.read_csv("../Outputs/labeled_df_binary.csv") \
        .drop(["gold_label_them"], axis=1)
    labels = labeled_data_df["gold_label_us"]
    features = labeled_data_df.loc[:, "season_games_to_now_us":"tackles_for_loss_prior_s_them"]
    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.15,
                                                                        random_state=0)
    rf_class = RandomForestClassifier()
    n_estimators = [int(x) for x in np.linspace(start=400, stop=1600, num=13)]
    max_features = ["sqrt", "log2"]
    max_depth = [int(x) for x in np.linspace(start=5, stop=50, num=25)]
    min_split = [int(x) for x in np.linspace(start=2, stop=20, num=10)]
    min_leaf = [int(x) for x in np.linspace(start=1, stop=21, num=11)]
    bootstrap = [True, False]
    hyperparameter_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_split,
                           'min_samples_leaf': min_leaf,
                           'bootstrap': bootstrap}
    rf_optimized = RandomizedSearchCV(estimator=rf_class, param_distributions=hyperparameter_grid, n_iter=200,
                                      cv=8, verbose=2, random_state=0, n_jobs=-1)
    rf_optimized.fit(train_data, train_labels)
    print(rf_optimized.best_params_)
    best_rf = rf_optimized.best_estimator_
    accuracy = best_rf.score(test_data, test_labels)
    print(f"Accuracy of best rf: {accuracy}")


# Experiment 12 tests whether the model can get better accuracy if the first n games of the season are excluded,
# based on the idea that some bettors may skip the first few games of each season in order to get a better idea
# of where each team stands
def experiment_12(is_classification: bool):
    GameCSVBuilder.build_except_first_n_games(2, is_classification)
    training_percent = 0.85
    batch_size = 128
    file_name = "../Outputs/labeled_data_except_2" + ("_binary.csv" if is_classification else ".csv")
    print("File name is "+file_name)
    loaders = NFL_Dataset.get_train_test_loaders(file_name, training_percent, batch_size)
    input_size = len(loaders[0].dataset[0][0])
    num_epochs = 20
    loss_function = nn.SmoothL1Loss(reduction='sum')
    if is_classification:
        loss_function = nn.BCELoss(reduction='sum')
    csv_output = "hid_layer_neurons,best_test_loss,best_pred_accuracy\n"
    for num_hid_layers in range(2, input_size + 1):
        nfl_mlp = NFL_NN_2.NFL_MLP(loaders, 3, [input_size, num_hid_layers, 1], nn.ReLU, is_classification)
        optimizer = torch.optim.Adam(nfl_mlp.parameters(), lr=0.0008)
        numpy_results = nfl_mlp.train_nn(loss_function, optimizer, num_epochs)
        best_test_loss = np.min(numpy_results[:, 1])
        best_prediction_rate = np.max(numpy_results[:, 2])
        csv_output += f"{num_hid_layers},{best_test_loss},{best_prediction_rate}\n"
    results_file_name = "../ExperimentResults/experiment_12_" + \
                        ("classification" if is_classification else "regression")
    test_results = open(results_file_name+".csv", "w")
    test_results.write(csv_output)
    test_results.close()


# This tests different configurations of inner layers with a 4-layer feed-forward neural network. It is
# the same as experiment_3 but uses the data that excludes the first two games of each season.
def experiment_13():
    GameCSVBuilder.build_except_first_n_games(2, True)
    training_percent = 0.85
    batch_size = 128
    file_name = "../Outputs/labeled_data_except_2_binary.csv"
    loaders = NFL_Dataset.get_train_test_loaders(file_name, training_percent, batch_size)
    input_size = len(loaders[0].dataset[0][0])
    num_epochs = 20
    loss_function = nn.BCELoss(reduction='sum')
    csv_output = "layer_2_neurons,layer_3_neurons,best_test_loss,best_pred_accuracy\n"
    for layer_2_neurons in range(2, input_size+1, 2):
        for layer_3_neurons in range(2, input_size+1, 2):
            print(f"Training NN with size {input_size}, {layer_2_neurons}, {layer_3_neurons}, 1")
            nfl_mlp = NFL_NN_2.NFL_MLP(loaders, 4, [input_size, layer_2_neurons, layer_3_neurons, 1], nn.ReLU, True)
            optimizer = torch.optim.Adam(nfl_mlp.parameters(), lr=0.0008)
            numpy_results = nfl_mlp.train_nn(loss_function, optimizer, num_epochs, False)
            best_test_loss = np.min(numpy_results[:, 1])
            best_prediction_rate = np.max(numpy_results[:, 2])
            csv_output += f"{layer_2_neurons},{layer_3_neurons},{best_test_loss},{best_prediction_rate}\n"
    test_results = open("../ExperimentResults/experiment_13.csv", "w")
    test_results.write(csv_output)
    test_results.close()
