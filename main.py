import random
import pandas as pd
import numpy as np
import sys

# set up
formats = ["t20", "odi", "test", "t10", "t5"]
format = random.sample(formats, 1)[0]
balls = 6
wickets = 10
players = 11
teams_list = ["india", "australia", "england", "pakistan"]
team_list = random.sample(teams_list, 2)
team_dataframes = {}
team_batters = {}
team_bowlers = {}
toss_choice = ["bat", "bowl"]

if format == "t20":
    overs = 20
    num_bowlers = 4
elif format == "odi":
    overs = 50
    num_bowlers = 5
elif format == "test":
    overs = 1000
    num_bowlers = 5
elif format == "t10":
    overs = 10
    num_bowlers = 2
elif format == "t5":
    overs = 5
    num_bowlers = 2


# functions
def get_bowlers(team_dataframe, num_bowlers):
    sorted_team = team_dataframe.sort_values(by="Bowling", ascending=False)
    top_bowlers = sorted_team.head(num_bowlers)
    return top_bowlers


def get_batters(team_dataframe):
    sorted_team = team_dataframe.sort_values(by="Batting", ascending=False)
    top_batters = sorted_team.head(players)
    return top_batters


def toss(team_list, toss_choice):
    toss_winner = random.choice(team_list)
    toss_choice = random.choice(toss_choice)
    return toss_winner, toss_choice


def urgency(runs_remaining, balls_remaining, wickets_remaining):
    max_run_rate = 36  # Maximum runs per over (6 balls) if each ball is a six
    required_run_rate = (runs_remaining / balls_remaining) * 6

    # Increase the urgency as the number of wickets decreases
    wicket_factor = (10 - wickets_remaining) / 10

    urgency = (required_run_rate / max_run_rate) * (1 + wicket_factor)

    return urgency


def ball_result(bowler, batter, urgency, chasing):
    batter_rating = batter["Batting"]
    bowler_rating = bowler["Bowling"].values[0]
    rating_diff = batter_rating - bowler_rating

    results = [0, 1, 2, 3, 4, 6, "out"]

    if rating_diff <= -10:
        base_weights = [
            0.3,
            0.2,
            0.1,
            0.1,
            0.1,
            0.05,
            0.15,
        ]  # bowler has significant advantage
    elif -10 < rating_diff <= 0:
        base_weights = [
            0.2,
            0.2,
            0.15,
            0.15,
            0.1,
            0.05,
            0.15,
        ]  # bowler has slight advantage
    elif 0 < rating_diff <= 10:
        base_weights = [
            0.1,
            0.2,
            0.2,
            0.2,
            0.15,
            0.1,
            0.05,
        ]  # batter has slight advantage
    else:
        base_weights = [
            0.05,
            0.15,
            0.2,
            0.2,
            0.2,
            0.15,
            0.05,
        ]  # batter has significant advantage

    if chasing == True:
        # Calculate the adjustment factor for weights based on urgency
        if urgency <= 0.5:
            # For low urgency, increase the chance of safer runs (1,2,3) and decrease the chance of 6 and 'out'
            adjustment_factors = [1.1, 1.2, 1.2, 1.2, 1.1, 0.9, 0.9]
        else:
            # For high urgency, increase the chance of 4, 6, and 'out'
            adjustment_factors = [0.9, 0.9, 0.9, 1.1, 1.2, 1.2, 1.2]

        # Adjust the weights based on the urgency
        weights = [
            base_weight * adjustment_factor
            for base_weight, adjustment_factor in zip(base_weights, adjustment_factors)
        ]

        # Normalize the weights so they sum to 1
        total_weight = sum(weights)
        weights = [weight / total_weight for weight in weights]

        result = random.choices(results, weights=weights, k=1)[0]

        return result

    else:
        total_weight = sum(base_weights)
        weights = [weight / total_weight for weight in base_weights]

        result = random.choices(results, weights=weights, k=1)[0]

        return result


def innings(batting_team, bowling_team, chasing, target):
    if chasing == False:
        wicket = 0
        over = 0
        ball = 0
        batters = batting_team.iloc[:players]  # get the first 11 batters
        bowlers = bowling_team.iloc[:num_bowlers]  # get the first num bowlers
        current_batter_index = 0
        run = 0
        runs = 0
        batter1 = batters.iloc[current_batter_index]
        batter2 = batters.iloc[(current_batter_index + 1) % players]
        batter_stats = {
            batter["Player"]: {"runs": 0, "balls_faced": 0}
            for _, batter in batters.iterrows()
        }
        bowler_stats = {
            bowler["Player"]: {"runs": 0, "overs_bowled": 0, "wickets": 0}
            for _, bowler in bowlers.iterrows()
        }

        while (wicket < wickets) and (over < overs):
            bowler_name = bowlers.sample().Player.values[0]
            bowler = bowlers.sample()
            print(
                f"Over: {over}, Score: {runs}-{wicket}, Bowler: {bowler_name}, CRR: {round(((runs/over) if over != 0 else 0),2)}"
            )
            while ball < balls:
                # The strike changes after each over or after taking an odd number of runs
                if ball == 0 or run % 2 != 0:
                    batter1, batter2 = batter2, batter1
                batter_on_strike = (
                    batter1  # The batter who is facing the current delivery
                )
                batter_stats[batter_on_strike["Player"]]["balls_faced"] += 1
                print(
                    f"Batter on strike: {batter_on_strike['Player']}"
                )
                result = ball_result(bowler, batter_on_strike, urgency=0, chasing=False)
                print(f"{over}.{ball} - {result}")
                if result == "out":
                    print(f"{batter_on_strike['Player']} is out")
                    wicket += 1
                    bowler_stats[bowler_name]["wickets"] += 1
                    if wicket == wickets:
                        break
                    current_batter_index = (current_batter_index + 1) % players
                    batter1 = batters.iloc[current_batter_index]  # New batter on strike
                else:
                    run = result
                    runs += result  # Update the run with the result of the ball
                    batter_stats[batter_on_strike["Player"]]["runs"] += result
                    bowler_stats[bowler_name]["runs"] += result
                ball += 1
            ball = 0
            bowler_stats[bowler_name]["overs_bowled"] += 1
            over += 1

        # print out the stats at the end
        print("\nBATTING STATS")
        for player, stats in batter_stats.items():
            print(f"{player}: {stats['runs']} ({stats['balls_faced']})")

        print(f"\nTotal: {runs}-{wicket}")

        print("\nBOWLING STATS")
        for player, stats in bowler_stats.items():
            print(
                f"{player}: {stats['wickets']}-{stats['runs']} ({stats['overs_bowled']})"
            )

        return runs

    else:
        wicket = 0
        over = 0
        ball = 0
        batters = batting_team.iloc[:players]  # get the first 11 batters
        bowlers = bowling_team.iloc[:num_bowlers]  # get the first num bowlers
        current_batter_index = 0
        runs = 0
        target_score = target
        batter1 = batters.iloc[current_batter_index]
        batter2 = batters.iloc[(current_batter_index + 1) % players]
        batter_stats = {
            batter["Player"]: {"runs": 0, "balls_faced": 0}
            for _, batter in batters.iterrows()
        }
        bowler_stats = {
            bowler["Player"]: {"runs": 0, "overs_bowled": 0, "wickets": 0}
            for _, bowler in bowlers.iterrows()
        }

        while (wicket < wickets) and (over < overs) and (runs <= target_score):
            bowler_name = bowlers.sample().Player.values[0]
            bowler = bowlers.sample()
            print(
                f"Over: {over}, Score: {runs}-{wicket}, Bowler: {bowler_name}, CRR: {round(((runs/over) if over != 0 else 0),2)}, RRR: {round(((target_score - runs)/(overs - over)),2)}, Runs needed: {target_score - runs} in {(overs*6) - (over*6)} balls"
            )
            while ball < balls:
                # The strike changes after each over or after taking an odd number of runs
                if ball == 0 or run % 2 != 0:
                    batter1, batter2 = batter2, batter1
                batter_on_strike = (
                    batter1  # The batter who is facing the current delivery
                )
                batter_stats[batter_on_strike["Player"]]["balls_faced"] += 1
                print(
                    f"Batter on strike: {batter_on_strike['Player']}"
                )
                urgency_factor = urgency(
                    (target_score - runs), (overs * 6) - (over * 6), (wickets - wicket)
                )
                result = ball_result(
                    bowler, batter_on_strike, urgency_factor, chasing=True
                )
                print(f"{over}.{ball} - {result}")
                if result == "out":
                    # print(f"{batter_on_strike['Player']} is out")
                    run = 0
                    wicket += 1
                    bowler_stats[bowler_name]["wickets"] += 1
                    if wicket == wickets:
                        break
                    current_batter_index = (current_batter_index + 1) % players
                    batter1 = batters.iloc[current_batter_index]  # New batter on strike
                else:
                    run = result
                    runs += result  # Update the run with the result of the ball
                    batter_stats[batter_on_strike["Player"]]["runs"] += result
                    bowler_stats[bowler_name]["runs"] += result
                    if runs > target_score:
                        break
                ball += 1
            ball = 0
            bowler_stats[bowler_name]["overs_bowled"] += 1
            over += 1

        # print out the stats at the end
        print("\nBATTING STATS")
        for player, stats in batter_stats.items():
            print(f"{player}: {stats['runs']} ({stats['balls_faced']})")

        print(f"\nTotal: {runs}-{wicket}")

        print("\nBOWLING STATS")
        for player, stats in bowler_stats.items():
            print(
                f"{player}: {stats['wickets']}-{stats['runs']} ({stats['overs_bowled']})"
            )

        balls_remaining = (overs * 6) - (over * 6)

        return runs, wicket, balls_remaining


def match(format, team_list, toss_choice):
    original_stdout = sys.stdout

    toss_winner = toss(team_list, toss_choice)
    home_team = toss_winner[0]
    away_team = team_list[0] if team_list[0] != home_team else team_list[1]

    with open(f'output/{home_team}_{away_team}_{format}.txt', 'w') as f:
        sys.stdout = f
        # print(home_team, away_team)
        print(f"{toss_winner[0].capitalize()} won the toss and elected to {toss_winner[1]} first")

        for team in team_list:
            team_dataframes[team] = pd.read_csv(f"{team}.csv")

        for team in team_list:
            team_dataframe = team_dataframes[team]
            team_bowlers[team] = get_bowlers(team_dataframe, num_bowlers)
            team_batters[team] = get_batters(team_dataframe)

        if toss_winner[1] == "bat":
            first_batting_team = team_batters[home_team].reset_index(drop=True)
            first_bowling_team = team_bowlers[away_team].reset_index(drop=True)
            second_batting_team = team_batters[away_team].reset_index(drop=True)
            second_bowling_team = team_bowlers[home_team].reset_index(drop=True)
            first_team = home_team
            chasing_team = away_team
        else:
            first_batting_team = team_batters[away_team].reset_index(drop=True)
            first_bowling_team = team_bowlers[home_team].reset_index(drop=True)
            second_batting_team = team_batters[home_team].reset_index(drop=True)
            second_bowling_team = team_bowlers[away_team].reset_index(drop=True)
            first_team = away_team
            chasing_team = home_team

        # print(first_batting_team, first_bowling_team)

        first_innings_runs = innings(first_batting_team, first_bowling_team, False, 0)
        print(
            f"{chasing_team.capitalize()} needs {first_innings_runs + 1} runs to win in {overs} overs at RR of {(first_innings_runs + 1) / overs}"
        )
        second_innings_runs = innings(
            second_batting_team, second_bowling_team, True, (first_innings_runs + 1)
        )
        if second_innings_runs[0] > first_innings_runs:
            print(
                f"\n{chasing_team.capitalize()} won by {10 - second_innings_runs[1]} wickets with {second_innings_runs[2]} balls remaining"
            )
        else:
            print(
                f"\n{first_team.capitalize()} won by {first_innings_runs - second_innings_runs[0]} runs"
            )
        
        sys.stdout = original_stdout


match(format, team_list, toss_choice)