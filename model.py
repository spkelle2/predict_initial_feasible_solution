import os

from cylp.cy import CyClpSimplex
import gurobi as gu

from schemas import mip_input_schema


def make_mps_file(dat: mip_input_schema.TicDat):

    # for my own sanity to make sure I made the input data as expected
    assert not mip_input_schema.find_foreign_key_failures(dat)
    assert not mip_input_schema.find_data_type_failures(dat)
    assert not mip_input_schema.find_data_row_failures(dat)

    p = mip_input_schema.create_full_parameters_dict(dat)

    # division map
    division = {team: [opponent for opponent, opponent_field in dat.teams.items()
                       if opponent_field['division'] == team_field['division']
                       and opponent != team] for team, team_field in dat.teams.items()}

    mdl = gu.Model()

    # enforce x to be 0 if any of home, away, or week are dummy values
    x = {(home, away, week): mdl.addVar(vtype=gu.GRB.BINARY, name=f'x_{home}_{away}_{week}')
         if home_field['type'] == away_field['type'] == week_field['type'] == 'active'
         else mdl.addVar(ub=0, name=f'x_{home}_{away}_{week}')
         for home, home_field in dat.teams.items() for away, away_field in dat.teams.items()
         for week, week_field in dat.weeks.items() if home != away}

    mdl.setObjective(sum(dat.revenue[home, away, week]['revenue'] * x[home, away, week]
                         for home, away, week in x), gu.GRB.MAXIMIZE)

    for team in dat.teams:
        for week in dat.weeks:

            # 1. each team plays at most one game each week, either at home or away
            mdl.addConstr(
                sum(x[team, away, week] for away in dat.teams if away != team) +
                sum(x[home, team, week] for home in dat.teams if home != team) <= 1,
                name=f'team_{team}_week_{week}'
            )

        # 2. each team must play a similar number of home and away games throughout the season
        mdl.addConstr(
            sum(sum(x[team, away, week] for away in dat.teams if away != team) -
                sum(x[home, team, week] for home in dat.teams if home != team)
                for week in dat.weeks) <= 1, name=f'team_{team}_balance_over'
        )
        mdl.addConstr(
            sum(sum(x[team, away, week] for away in dat.teams if away != team) -
                sum(x[home, team, week] for home in dat.teams if home != team)
                for week in dat.weeks) >= -1, name=f'team_{team}_balance_under'
        )

        # 3. each team must play as many games within their division as outside
        mdl.addConstr(
            sum(sum(x[team, away, week] for away in dat.teams if away != team and away in division[team]) +
                sum(x[home, team, week] for home in dat.teams if home != team and home in division[team]) -
                sum(x[team, away, week] for away in dat.teams if away != team and away not in division[team]) -
                sum(x[home, team, week] for home in dat.teams if home != team and home not in division[team])
                for week in dat.weeks) >= 0, name=f'team_{team}_division'
        )

    mdl.update()

    file_base = f'models/schedule_{p["problem ID"]}'
    mdl.write(f'{file_base}.lp')

    # for some reason CyLP cannot read standard .mps files, so read in a .lp
    # file and let CyLP write the .mps file in a format it can understand
    lp = CyClpSimplex()
    lp.readLp(f'{file_base}.lp')
    lp.writeMps(f'{file_base}.mps', objSense=1)
    os.remove(f'{file_base}.lp')
