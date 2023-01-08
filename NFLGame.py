# Class NFLGame calculates the stats of each game given the large CSV of all NFL plays from 2009 to 2018
import pandas as pd


class NFLGame:

    def __init__(self, df: pd.DataFrame, team: str, game_id: str):
        self.df = df
        self.off_df = df.loc[(df["posteam"] == team) & (df["play_type"] != "no_play")]
        self.def_df = df.loc[(df["defteam"] == team) & (df["play_type"] != "no_play")]
        self.team = team
        self.game_id = game_id
        self.game_date = int(game_id[0: -2]) if type(game_id) == str else self.game_id // 100
        self.season = self.get_season()
        self.is_home_team = 1.0 if self.df["home_team"].values[0] == self.team else 0.0

    # Assumes date is int in form YYYYMMDD
    def get_season(self):
        date = self.game_date
        year = date//10000
        month = (date//100) % 100
        if month > 5:
            return year
        else:
            return year - 1

    def __str__(self):
        result = f"Date: {self.game_id} \nTeam: {self.team}\n\n"
        return result

    def get_all_stats_dict(self):
        point_differential = self.get_point_differential()
        result = {"game_id": self.game_id,
                  "game_date": self.game_date,
                  "team": self.team,
                  "season": self.season,
                  "point_dif": point_differential,
                  "is_home_team": self.is_home_team
                  }
        off_stat_dict = self.extract_offensive_stats()
        for key in off_stat_dict:
            result[key] = off_stat_dict[key]
        def_stat_dict = self.extract_defensive_stats()
        for key in def_stat_dict:
            result[key] = def_stat_dict[key]
        return result

    def print_all_stats(self):
        print(f"============All calculated stats for {self.team}, gameID: {self.game_id}===========")
        stats = self.get_all_stats_dict()
        start_row = False
        for key in stats:
            if not start_row:
                print(f"  {key:20}: {stats[key]:8}  |  ", end="")
            else:
                print(f"  {key:20}: {stats[key]:8}")
            start_row = not start_row
        print()

    def get_point_differential(self):
        last_play_index = self.df.last_valid_index()
        home_score = self.df.loc[last_play_index, "total_home_score"]
        away_score = self.df.loc[last_play_index, "total_away_score"]
        if self.is_home_team == 1:
            return home_score - away_score
        else:
            return away_score - home_score

    # ==== General class stuff and utility functions above this line, offensive stat calculations below====
    def get_off_sum(self, column_name: str):
        result = self.off_df[column_name].astype(float).sum()
        # print(f"sum of column name {column_name}: {result}")
        return result

    def pass_yards_attempts(self):
        df = self.off_df[self.off_df["play_type"] == "pass"]
        return df["yards_gained"].astype(float).sum(), df.shape[0]

    def rush_yards_attempts(self):
        df = self.off_df[self.off_df["play_type"] == "run"]
        return df["yards_gained"].astype(float).sum(), df.shape[0]

    def get_off_penalties(self):
        df = self.df[(self.df["penalty_team"] == self.team) & (self.df["posteam"] == self.team)]
        return df["penalty_yards"].astype(float).sum(), df.shape[0]

    def get_fg_made_yards(self):
        fg_made = len(self.off_df[self.off_df["field_goal_result"] == "made"])
        total_attempted_yards = 0
        # below checks if any field goals were attempted, keeps code from crashing
        # if no field goals made
        if len(self.off_df["field_goal_result"].value_counts()) > 0:
            df = self.off_df[self.off_df["field_goal_attempt"] == 1]
            total_attempted_yards = df["kick_distance"].astype(float).sum()
        return fg_made, total_attempted_yards

    def get_total_start_pos(self):
        cur_drive = 0
        result = 0
        add_next = False
        for cur_index in self.off_df.index:
            if add_next:
                result += self.off_df["yardline_100"][cur_index]
                add_next = False
            if cur_drive == self.off_df["drive"][cur_index]:
                pass
            else:
                cur_drive = self.off_df["drive"][cur_index]
                # if kickoff, adds yardline from subsequent play to reflect where offense
                # starts playing after kickoff
                if self.off_df["play_type"][cur_index] == "kickoff":
                    add_next = True
                else:
                    result += self.off_df["yardline_100"][cur_index]
        return result

    def get_time_possession(self):
        # This approximates TOP by subtracting the time at the start of the last play from that
        # at the start of the first play. Time from END of this drive to START of next drive(for
        # other team) is not counted.
        result = 0
        off_drive_start = self.off_df.groupby("drive")["game_seconds_remaining"].max()
        off_drive_end = self.off_df.groupby("drive")["game_seconds_remaining"].min()
        for start, end in zip(off_drive_start, off_drive_end):
            result += start - end
        return result

    def get_sacks_allowed_sackyards(self):
        df = self.off_df[self.off_df["sack"] == 1]
        df = df[df["yards_gained"] <= 0]
        return df.shape[0], df["yards_gained"].astype(float).sum()

    # returns a vector of all offensive stats.
    # Call print offensive stats to see printed out with labels
    def extract_offensive_stats(self):
        # number of offensive off_drives
        off_drives = len(self.off_df["drive"].value_counts())
        # total starting field position
        off_total_start_pos = self.get_total_start_pos()
        # completed passes
        completed_passes = self.get_off_sum("complete_pass")
        # total pass yards
        net_passing_yards, passes_attempted = self.pass_yards_attempts()
        # air yards
        air_yards = self.get_off_sum("air_yards")
        # yards after catch
        yards_after_catch = self.get_off_sum("yards_after_catch")
        # passing touchdowns
        pass_td = self.get_off_sum("pass_touchdown")
        # Interceptions
        interceptions_thrown = self.get_off_sum("interception")
        # Fumbles
        fumbles_made = self.get_off_sum("fumble")
        # Fumbles lost
        fumbles_lost = self.get_off_sum("fumble_lost")
        # rushing attempts and rushing yards
        off_total_rushing, rushes_attempted = self.rush_yards_attempts()
        # rushing touchdowns
        rush_td = self.get_off_sum("rush_touchdown")
        # Number of penalties and penalty yards
        off_pen_yards, off_penalties = self.get_off_penalties()
        # time of possession.
        time_possession = self.get_time_possession()
        # field goal attempts
        fg_attempts = self.get_off_sum("field_goal_attempt")
        # fg made, total yards from which field goals attempted
        fg_made, fg_attempt_yards = self.get_fg_made_yards()
        # sacks allowed
        times_sacked, sacked_yards_lost = self.get_sacks_allowed_sackyards()
        # QB hits allowed. NOTE: Not available on other soruces, hard to cross-verify
        qb_was_hit = self.get_off_sum("qb_hit")
        # Number   of first downs
        first_downs_earned = self.get_off_sum("first_down_rush") + self.get_off_sum("first_down_pass")

        # stats_list = [off_drives, off_total_start_pos, completed_passes, net_passing_yards,
        #               passes_attempted, air_yards, yards_after_catch, pass_td, interceptions_thrown,
        #               fumbles_made, fumbles_lost, off_total_rushing, rushes_attempted, rush_td,
        #               off_pen_yards, off_penalties, time_possession, fg_attempts, fg_made,
        #               fg_attempt_yards, times_sacked, qb_was_hit, first_downs_earned, sacked_yards_lost]

        # return np.asarray(stats_list, dtype=np.float64)

        off_stats_dict = {
            "off_drives": off_drives,
            "off_total_start_pos": off_total_start_pos,
            "completed_passes": completed_passes,
            "net_passing_yards": net_passing_yards,
            "passes_attempted": passes_attempted,
            "air_yards": air_yards,
            "yards_after_catch": yards_after_catch,
            "pass_td": pass_td,
            "interceptions_thrown": interceptions_thrown,
            "fumbles_made": fumbles_made,
            "fumbles_lost": fumbles_lost,
            "off_total_rushing": off_total_rushing,
            "rushes_attempted": rushes_attempted,
            "rush_td": rush_td,
            "off_pen_yards": off_pen_yards,
            "off_penalties": off_penalties,
            "time_possession": time_possession,
            "fg_attempts": fg_attempts,
            "fg_made": fg_made,
            "fg_attempt_yards": fg_attempt_yards,
            "times_sacked": times_sacked,
            "qb_was_hit": qb_was_hit,
            "first_downs_earned": first_downs_earned,
            "sacked_yards_lost": sacked_yards_lost
        }
        return off_stats_dict

    def print_offensive_stats(self):
        print(f"==============Offensive stats for {self.team}, gameID: {self.game_id}==============")
        stats = self.extract_offensive_stats()
        start_row = False
        for key in stats:
            if not start_row:
                print(f"  {key:20}: {stats[key]:8}  |  ", end="")
            else:
                print(f"  {key:20}: {stats[key]:8}")
            start_row = not start_row
        print()

    # ================ offensive stat calcluations above this line, defensive stats below ===============

    def get_def_sum(self, column_name: str):
        result = self.def_df[column_name].astype(float).sum()
        # print(f"sum of column name {column_name}: {result}")
        return result

    def get_opp_fg_attempt_total_yards(self):
        total_attempted_yards = 0
        # below checks if any field goals were attempted, keeps code from crashing
        # if no field goals made
        if len(self.def_df["field_goal_result"].value_counts()) > 0:
            df = self.def_df[self.def_df["field_goal_attempt"] == 1]
            total_attempted_yards = df["kick_distance"].astype(float).sum()
        return total_attempted_yards

    def get_opponent_total_start_pos(self):
        cur_drive = 0
        result = 0
        add_next = False
        for cur_index in self.def_df.index:
            if add_next:
                result += self.def_df["yardline_100"][cur_index]
                add_next = False
            if cur_drive == self.def_df["drive"][cur_index]:
                pass
            else:
                cur_drive = self.def_df["drive"][cur_index]
                # if kickoff, adds yardline from subsequent play to reflect where offense
                # starts playing after kickoff
                if self.def_df["play_type"][cur_index] == "kickoff":
                    add_next = True
                else:
                    result += self.def_df["yardline_100"][cur_index]
        return result

    def get_def_penalties(self):
        df = self.df[(self.df["penalty_team"] == self.team) & (self.df["defteam"] == self.team)]
        return df["penalty_yards"].astype(float).sum(), df.shape[0]

    def pass_allowed_yards_attempts(self):
        df = self.def_df[self.def_df["play_type"] == "pass"]
        return df["yards_gained"].astype(float).sum(), df.shape[0]

    def rush_allowed_yards_attempts(self):
        df = self.def_df[self.def_df["play_type"] == "run"]
        return df["yards_gained"].astype(float).sum(), df.shape[0]

    def get_sacks_sackyards(self):
        df = self.def_df[self.def_df["sack"] == 1]
        # gives warning, unclear why, have tried fixing by using .loc but still gives warning
        df["yards_gained"] = df["yards_gained"].astype(float)
        df = df[df["yards_gained"] <= 0]
        return df.shape[0], df["yards_gained"].astype(float).sum()

    def extract_defensive_stats(self):
        pass_td_allowed = self.get_def_sum("pass_touchdown")
        rush_td_allowed = self.get_def_sum("rush_touchdown")
        opp_fg_attempts = self.get_def_sum("field_goal_attempt")
        opp_fg_attempt_yards = self.get_opp_fg_attempt_total_yards()
        interceptions_caught = self.get_def_sum("interception")
        forced_fumbles = self.get_def_sum("fumble_forced")
        def_penalty_yards, def_num_penalties = self.get_def_penalties()
        first_downs_allowed = self.get_def_sum("first_down_rush") + self.get_def_sum("first_down_pass")
        def_drives = len(self.def_df["drive"].value_counts())
        num_def_plays = self.def_df.shape[0]
        oppo_start_pos_total = self.get_opponent_total_start_pos()
        # p attempts is 1-2 off pro-football-reference.com numbers, both here and in offense
        p_yards_allowed, opponent_p_attempts = self.pass_allowed_yards_attempts()
        # rush attempts slightly off 3rd party numbers, but this appears to be because
        # the given dataset counts kneel-downs as 'no play' whereas others count
        # them as run/rush
        r_yards_allowed, opponent_r_attempts = self.rush_allowed_yards_attempts()
        pick_6s = self.get_picks_6s()
        hit_their_qb = self.get_def_sum("qb_hit")
        sacked_their_qb, sack_yards = self.get_sacks_sackyards()
        tackles_for_loss = self.get_def_sum("tackled_for_loss")

        def_stats_dict = {
            "opp_fg_attempts": opp_fg_attempts,
            "opp_fg_attempt_yards": opp_fg_attempt_yards,
            "interceptions_caught": interceptions_caught,
            "forced_fumbles": forced_fumbles,
            "def_penalty_yards": def_penalty_yards,
            "def_num_penalties": def_num_penalties,
            "first_downs_allowed": first_downs_allowed,
            "def_drives": def_drives,
            "num_def_plays": num_def_plays,
            "oppo_start_pos_total": oppo_start_pos_total,
            "p_yards_allowed": p_yards_allowed,
            "opponent_p_attempts": opponent_p_attempts,
            "pass_td_allowed": pass_td_allowed,
            "r_yards_allowed": r_yards_allowed,
            "opponent_r_attempts": opponent_r_attempts,
            "rush_td_allowed": rush_td_allowed,
            "pick_6s": pick_6s,
            "hit_their_qb": hit_their_qb,
            "sacked_their_qb": sacked_their_qb,
            "sack_yards": sack_yards,
            "tackles_for_loss": tackles_for_loss}

        return def_stats_dict

    def print_defensive_stats(self):
        print(f"==============Defensive stats for {self.team}, gameID: {self.game_id}==============")
        stats = self.extract_defensive_stats()
        start_row = False
        for key in stats:
            if not start_row:
                print(f"  {key:20}: {stats[key]:8}  |  ", end="")
            else:
                print(f"  {key:20}: {stats[key]:8}")
            start_row = not start_row
        print()

    def get_picks_6s(self):
        df = self.def_df[(self.def_df["interception"] == 1) & (self.def_df["touchdown"] == 1)]
        pick_6s = df.shape[0]
        return pick_6s
