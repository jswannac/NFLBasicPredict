# Functions in this class build numpy arrays for the neural network
# using the per-game data generated in NFLGame.py. There are also functions
# to build pandas dataframes for use in sklearn

import pandas as pd
from NFLGame import NFLGame
import numpy as np


def build_per_game_csv(df: pd.DataFrame):
    game_id_list = df["game_id"].unique()
    games_added = 0
    list_of_games = []
    for game_id in game_id_list:
        current_game_df = df.loc[df["game_id"] == game_id]
        home_team = current_game_df["home_team"].values[0]
        away_team = current_game_df["away_team"].values[0]
        list_of_games.append(NFLGame(current_game_df, home_team, game_id).get_all_stats_dict())
        list_of_games.append(NFLGame(current_game_df, away_team, game_id).get_all_stats_dict())
        games_added += 2
        print(f"Number of games added now: {games_added}")
    result = pd.DataFrame.from_records(list_of_games)
    update_old_team_names(result)
    result.to_csv("../Outputs/per_game_stats.csv", index=False)
    return result


# takes in DataFrame with per-game stats and calculates per-season stats
def build_per_team_per_seasons_stats(df: pd.DataFrame):
    df = df.drop(["game_id", "game_date", "is_home_team"], axis=1)
    result = df.groupby(["season", "team"]).mean()
    # below two lines get rid of the groups by outputting to csv then reading back in
    result.to_csv("../Outputs/per_team_per_season_stats.csv")
    result = pd.read_csv("../Outputs/per_team_per_season_stats.csv")
    # append season averages from 2008, which aren't included in the big CSV and were taken/adapted from
    # pro-football-reference.com
    season_data_2008 = pd.read_csv("../ProFootball-Reference_2008_data/2008_stats_right_format.csv")
    result = pd.concat([season_data_2008, result])
    result.to_csv("../Outputs/per_team_per_season_stats.csv", index=False)


# works in place, for now updates STL and SD, can add OAK->LV in future if needed
# also fixes weird mis-spelling in original file, JAX/JAC both used for Jacksonville,
# JAX used for both here.
def update_old_team_names(df: pd.DataFrame):
    df['team'].replace(to_replace="STL", value="LA", inplace=True)
    df['team'].replace(to_replace="SD", value="LAC", inplace=True)
    df['team'].replace(to_replace="JAC", value="JAX", inplace=True)


def build_game_up_to_stats(df: pd.DataFrame):
    result = []
    for index in df.index:
        game_id, season, date, team, is_home_team = \
            df.loc[index, ["game_id", "season", "game_date", "team", "is_home_team"]]
        original_point_dif = df.loc[index, "point_dif"]
        # get all games up to that point in that season, for that team
        season_games_to_now = df[(df["season"] == season) & (df["game_date"] < date) & (df["team"] == team)]
        num_games_to_now = season_games_to_now.shape[0]
        season_games_to_now = season_games_to_now\
            .drop(["game_id", "season", "game_date", "team", "is_home_team"], axis=1)\
            .mean(numeric_only=True)
        # season_games_to_now[["game_id", "season", "game_date", "team"]] = game_id, season, date, team
        data_added = pd.Series([original_point_dif, game_id, num_games_to_now, season, date, team, is_home_team],
                               ["gold_label", "game_id", "season_games_to_now", "season", "game_date",
                                "team", "is_home_team"])
        data_added = pd.concat((data_added, season_games_to_now))
        result.append(data_added)
    output_df = pd.DataFrame(result)
    output_df.to_csv("../Outputs/game_up_to_now_stats.csv", index=False)


def fill_1st_game_of_season(game_datum_series: pd.Series, start_label, end_label, prior_season_series,
                            fill_from_prior_season=False):
    if fill_from_prior_season:
        game_datum_series[start_label: end_label] = prior_season_series[start_label: end_label]
    else:
        game_datum_series[start_label:  end_label] = 0.0


def get_prior_season_series(per_season_df: pd.DataFrame, season: int, team: str):
    prior_season = int(season) - 1
    prior_season_df = per_season_df[(per_season_df["season"] == prior_season) & (per_season_df["team"] == team)] \
        .drop(["season", "team"], axis=1)
    return prior_season_df.loc[prior_season_df.last_valid_index()]


# this adds label suffixes to this season's and last season's statistics but avoids suffixing
# stuff like game_id or team, which aren't duplicated in this and last season
def handle_suffixes(game_datum_series: pd.Series, prior_season_series: pd.Series):
    game_datum_series = pd.concat((game_datum_series[:"is_home_team"],
                                   game_datum_series["point_dif":].add_suffix("_uptonow")))
    prior_season_series = prior_season_series.add_suffix("_prior_s")
    return pd.concat((game_datum_series, prior_season_series))


def build_per_game_per_team_df(up_to_now_df: pd.DataFrame, per_season_df: pd.DataFrame, fill_from_prior_season=False):
    list_of_labeled_datum = []
    for _, game_datum_series in up_to_now_df.iterrows():
        season, team = game_datum_series["season"], game_datum_series["team"]
        prior_season_series = get_prior_season_series(per_season_df, season, team)
        if game_datum_series["season_games_to_now"] == 0:
            fill_1st_game_of_season(game_datum_series, "point_dif", "tackles_for_loss",
                                    prior_season_series, fill_from_prior_season)
        full_game_datum = handle_suffixes(game_datum_series, prior_season_series)
        list_of_labeled_datum.append(full_game_datum)
    result_df = pd.DataFrame(list_of_labeled_datum)
    result_df.to_csv("../Outputs/game_inputs_partly_built.csv", index=False)
    return result_df


# parameter is return value from build_per_game_per_team_df
# we build a df that has statlines for hometeam, away team and away team, home team for each game
# this assumes the df has statlines per team for each game in consecutive rows
def build_labeled_data_numpy(games_df: pd.DataFrame, output_file_name: str, is_binary_classifier=False):
    list_of_labeled_datum = []
    is_odd_row = False
    prior_statline = None
    games_df = games_df.drop(["season",	"game_date", "team"], axis=1)
    for _, game_datum in games_df.iterrows():
        game_team_statline = game_datum.to_numpy()
        if is_odd_row:
            list_of_labeled_datum.append(np.concatenate((game_team_statline, prior_statline[2:])))
            list_of_labeled_datum.append(np.concatenate((prior_statline, game_team_statline[2:])))
        is_odd_row = not is_odd_row
        prior_statline = game_team_statline
    if is_binary_classifier:
        labelled_scores_to_win_loss(list_of_labeled_datum)
        output_file_name += "_binary"
    np.savetxt(output_file_name+".csv", list_of_labeled_datum, delimiter=',')
    return list_of_labeled_datum


# Changes a numpy with stats and a score as the label (in column 0) into a numpy for classification
# with wins labeled as 1 for the given team and everything else as 0
# modifies in-place, no return value
def labelled_scores_to_win_loss(np_labeled_data):
    for row in np_labeled_data:
        row[0] = 1.0 if row[0] > 0.0 else 0.0


# Full data pipeline to build 188 dimensional inputs from original data set (of giant csv)
def build_188_dim_data(is_binary_classifier: bool):
    df = pd.read_csv("../Giant700kRowsAndSubsets/GiantCSV2009-2018Trimmed.csv_cleaned.csv")
    build_per_game_csv(df)
    df = pd.read_csv("../Outputs/per_game_stats.csv")
    build_per_team_per_seasons_stats(df)
    df = pd.read_csv("../Outputs/per_game_stats.csv")
    build_game_up_to_stats(df)
    per_season_df = pd.read_csv("../Outputs/per_team_per_season_stats.csv")
    up_to_game_df = pd.read_csv("../Outputs/game_up_to_now_stats.csv")
    games_df = build_per_game_per_team_df(up_to_game_df, per_season_df, True)
    build_labeled_data_numpy(games_df, "../Outputs/labeled_data", is_binary_classifier)


# Builds a smaller vector of stats, chosen as those less important on listing in RF model
def build_smaller_dim_data(is_binary_classifier: bool):
    per_season_df = pd.read_csv("../Outputs/per_team_per_season_stats.csv")
    up_to_game_df = pd.read_csv("../Outputs/game_up_to_now_stats.csv")
    games_df = build_per_game_per_team_df(up_to_game_df, per_season_df, True)
    to_drop = ["times_sacked_prior_s",
                    "time_possession_prior_s",
                    "tackles_for_loss_prior_s",
                    "season_games_to_now",
                    "sacked_yards_lost_prior_s",
                    "sacked_their_qb_prior_s",
                    "rushes_attempted_prior_s",
                    "rush_td_prior_s",
                    "rush_td_allowed_uptonow",
                    "rush_td_allowed_prior_s",
                    "qb_was_hit_prior_s",
                    "pick_6s_uptonow",
                    "pick_6s_prior_s",
                    "passes_attempted_prior_s",
                    "pass_td_prior_s",
                    "pass_td_allowed_prior_s",
                    "p_yards_allowed_prior_s",
                    "opponent_p_attempts_prior_s",
                    "oppo_start_pos_total_uptonow",
                    "oppo_start_pos_total_prior_s",
                    "opp_fg_attempts_uptonow",
                    "opp_fg_attempts_prior_s",
                    "opp_fg_attempt_yards_prior_s",
                    "off_total_rushing_prior_s",
                    "off_penalties_uptonow",
                    "off_penalties_prior_s",
                    "off_pen_yards_prior_s",
                    "off_drives_uptonow",
                    "off_drives_prior_s",
                    "num_def_plays_uptonow",
                    "num_def_plays_prior_s",
                    "interceptions_thrown_prior_s",
                    "interceptions_caught_uptonow",
                    "hit_their_qb_prior_s",
                    "fumbles_made_prior_s",
                    "fumbles_lost_uptonow",
                    "fumbles_lost_prior_s",
                    "forced_fumbles_uptonow",
                    "forced_fumbles_prior_s",
                    "fg_made_prior_s",
                    "fg_attempts_uptonow",
                    "fg_attempts_prior_s",
                    "fg_attempt_yards_prior_s",
                    "def_num_penalties_uptonow",
                    "def_num_penalties_prior_s",
                    "def_drives_uptonow",
                    "def_drives_prior_s",
                    "completed_passes_uptonow",
                    "completed_passes_prior_s",
                    "air_yards_prior_s"]
    games_df.drop(to_drop, axis=1, inplace=True)
    build_labeled_data_numpy(games_df, "../Outputs/labeled_data_small", is_binary_classifier)


def build_df_for_sklearn(file_name: str, is_binary_classifier: bool):
    per_season_df = pd.read_csv("../Outputs/per_team_per_season_stats.csv")
    up_to_game_df = pd.read_csv("../Outputs/game_up_to_now_stats.csv")
    games_df = build_per_game_per_team_df(up_to_game_df, per_season_df)
#    games_df = pd.read_csv("../Outputs/game_inputs_partly_built.csv")
    result = pd.DataFrame()
    is_odd_row = False
    prior_row = None
    games_df = games_df.drop(["season",	"game_date", "team"], axis=1)
    for _, game_datum in games_df.iterrows():
        current_row = game_datum
        if is_odd_row:
            new_row_1 = pd.concat((prior_row.add_suffix("_us"), current_row.add_suffix("_them")))
            new_row_2 = pd.concat((current_row.add_suffix("_us"), prior_row.add_suffix("_them")))
            new_df = pd.concat((new_row_1, new_row_2), axis=1).transpose()\
                .drop(["game_id_them", "season_games_to_now_them"], axis=1)
            result = pd.concat((result, new_df), ignore_index=True)
        prior_row = current_row
        is_odd_row = not is_odd_row
    if is_binary_classifier:
        file_name += "_binary"
        result["gold_label_us"] = result["gold_label_us"].map(lambda a: 0.0 if a <= 0.0 else 1.0)
    result.to_csv(file_name+".csv", index=False)


# This is mostly copy/pasted from above, could use a refactor
def build_weighted_df_for_sklearn(weighted_games_df: pd.DataFrame, file_name: str, is_binary_classifier: bool):
    result = pd.DataFrame()
    is_odd_row = False
    prior_row = None
    games_df = weighted_games_df.drop(["season", "game_date", "team"], axis=1)
    for _, game_datum in games_df.iterrows():
        current_row = game_datum
        if is_odd_row:
            new_row_1 = pd.concat((prior_row.add_suffix("_us"), current_row.add_suffix("_them")))
            new_row_2 = pd.concat((current_row.add_suffix("_us"), prior_row.add_suffix("_them")))
            new_df = pd.concat((new_row_1, new_row_2), axis=1).transpose()\
                .drop(["game_id_them"], axis=1)
            result = pd.concat((result, new_df), ignore_index=True)
        prior_row = current_row
        is_odd_row = not is_odd_row
    if is_binary_classifier:
        file_name += "_binary"
        result["gold_label_us"] = result["gold_label_us"].map(lambda a: 0.0 if a <= 0.0 else 1.0)
    result.to_csv(file_name+".csv", index=False)


def calc_weighted_game(prior_game_stats: pd.Series, two_games_prior_stats: pd.Series, alpha: float):
    return prior_game_stats.map(lambda x: x*(1-alpha)) + \
        two_games_prior_stats.map(lambda x: x*alpha)


def build_weighted_games(all_games_df: pd.DataFrame, per_team_per_season_df: pd.DataFrame, season: int,
                         team: str, alpha: float):
    games = []
    relevant_season_team_df = all_games_df[(all_games_df["team"] == team) &
                                           (all_games_df["season"] == season)]\
        .sort_values(by=["game_date"])
    # holds calculation of weighted games for one season and one team, fix starting values later
    prior_game_stats = get_prior_season_series(per_team_per_season_df, season, team)
    two_games_prior_stats = prior_game_stats
    for _, row in relevant_season_team_df.iterrows():
        row_info = row[["point_dif", "team", "season", "game_id", "game_date", "is_home_team"]]\
            .rename({"point_dif": "gold_label"})
        current_game = pd.concat([row_info, calc_weighted_game(prior_game_stats, two_games_prior_stats, alpha)])
        # need to add gameID and label back on
        games.append(current_game)
        two_games_prior_stats = prior_game_stats
        prior_game_stats = row.drop(["team", "season", "game_id", "game_date", "is_home_team"])
    return games


def build_weighted_game_vectors(all_games_df: pd.DataFrame, per_team_per_season_df: pd.DataFrame, alpha: float):
    weighted_games = []
    for _, row in per_team_per_season_df.iterrows():
        team, season = row["team"], row["season"]
        if season != 2008:
            weighted_games.extend(build_weighted_games(all_games_df, per_team_per_season_df, season, team, alpha))
    weighted_games = pd.DataFrame(weighted_games).sort_values(by=["game_id"])
    weighted_games.to_csv("weighted_games_temp.csv", index=False)
    return weighted_games


# This builds the data but excludes the first n games from a given season
def build_except_first_n_games(n_games: int, is_binary_classifier: bool):
    if n_games > 15 or n_games < 1:
        print(f"Error filtering first {n_games} in build_except_first_n_games: use a value between 1 and 15 inclusive")
        return
    games_df = pd.read_csv("../Outputs/game_up_to_now_stats.csv")
    games_df = games_df.loc[games_df["season_games_to_now"] >= n_games].drop(["pick_6s"], axis=1)
    # pick 6s too rare/random, does not do much, better to drop it
    build_labeled_data_numpy(games_df, "../Outputs/labeled_data_except_2", is_binary_classifier)
    build_weighted_df_for_sklearn(games_df, "../Outputs/labeled_df_except_2", is_binary_classifier)
