from math import ceil
import numpy as np
import os
import random
import shutil

from model import make_mps_file
from schemas import mip_input_schema


def make_models(league_sizes: list, season_lengths: list, num_intances: int):

    shutil.rmtree('models', ignore_errors=True)
    os.mkdir('models')

    for problem_id in range(num_intances):

        # make metadata
        max_teams = max(league_sizes)
        max_weeks = max(season_lengths)
        num_teams = random.choice(league_sizes)
        num_divisions = np.random.randint(1, num_teams/2)
        num_weeks = random.choice(season_lengths)
        # make sure all teams, even dummies, get a division
        div_list = (list(range(num_divisions))*ceil(max_teams/num_divisions))[:max_teams]

        # make tables
        # note: team and week are less than num_team and num_week since we index from 0
        teams = [{'team': str(team), 'division': str(division),
                  'type': 'active' if team < num_teams else 'dummy'}
                 for team, division in enumerate(div_list)]
        weeks = [{'week': str(week), 'type': 'active' if week < num_weeks else 'dummy'}
                 for week in range(max_weeks)]
        revenue = [{'home': str(home_team), 'away': str(away_team), 'week': str(week),
                    'revenue': round(np.random.uniform(10), 3) if home_team < num_teams
                    and away_team < num_teams and week < num_weeks else 0}
                   for home_team in range(max_teams) for away_team in range(max_teams)
                   for week in range(max_weeks) if home_team != away_team]
        parameters = [{'key': 'problem ID', 'value': problem_id}]

        dat = mip_input_schema.TicDat(teams=teams, weeks=weeks, revenue=revenue, parameters=parameters)

        make_mps_file(dat)


if __name__ == '__main__':
    make_models([4, 5, 6], [6, 7, 8], 100)