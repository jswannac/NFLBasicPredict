# Test functions used in development of NFL prediction class

import pandas as pd
from NFLGame import NFLGame
import GameCSVBuilder
import DataCleaner as dClean
import NFL_NN


def test_small_dataset_NN():
    NFL_NN.run_neural_network("../Outputs/easy_2game_dataset.csv")


def test_basic_stats_4games():
    df = pd.read_csv("../Giant700kRowsAndSubsets/ManyPenaltiesInOneGame.csv")
    current_game = df.loc[df["game_id"] == 2017121711]
    team = "DAL"
    # only get stats from rows in which this team is offense, filters out time-outs
    test_game = NFLGame(current_game, team, "2017121711")
    test_game.print_all_stats()
    team = "OAK"
    test_game = NFLGame(current_game, team, "2017121711")
    test_game.print_all_stats()
    df = pd.read_csv("../Giant700kRowsAndSubsets/GameWithPick6.csv")
    current_game = df.loc[df["game_id"] == 2017102600]
    team = "BAL"
    test_game = NFLGame(current_game, team, "2017102600")
    test_game.print_all_stats()
    team = "MIA"
    test_game = NFLGame(current_game, team, "2017102600")
    test_game.print_all_stats()


def test_point_differential():
    df = pd.read_csv("../Giant700kRowsAndSubsets/ManyPenaltiesInOneGame.csv")
    current_game = df.loc[df["game_id"] == 2017121711]
    team = "DAL"
    # only get stats from rows in which this team is offense, filters out time-outs
    test_game = NFLGame(current_game, team, "2017121711")
    test_game.get_point_differential()


def test_data_cleaner():
    dClean.data_clean_controller("../Giant700kRowsAndSubsets/GiantCSV2009-2018Trimmed.csv")


def test_csv_builder():
    df = pd.read_csv("../Giant700kRowsAndSubsets/2Season2TeamTest.csv_cleaned.csv")
    GameCSVBuilder.build_per_game_csv(df)


def test_per_season_stats():
    df = pd.read_csv("../Outputs/per_game_stats.csv")
    GameCSVBuilder.build_per_team_per_seasons_stats(df)


def test_per_game_up_to_stats():
    df = pd.read_csv("../Outputs/per_game_stats.csv")
    GameCSVBuilder.build_game_up_to_stats(df)

def test_build_labeled_data():
    per_season_df = pd.read_csv("../Outputs/per_team_per_season_stats.csv")
    up_to_game_df = pd.read_csv("../Outputs/game_up_to_now_stats.csv")
    GameCSVBuilder.build_per_game_per_team_df(up_to_game_df, per_season_df)
