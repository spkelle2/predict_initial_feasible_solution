from ticdat import TicDatFactory

mip_input_schema = TicDatFactory(
    parameters=[["key"], ["value"]],
    teams=[['team'], ['division', 'type']],
    weeks=[['week'], ['type']],
    revenue=[['home', 'away', 'week'], ['revenue']]
)

mip_solution_schema = TicDatFactory(
    metadata=[['key'], ['value']],
    schedule=[[], ['home', 'away', 'week']]
)

mip_input_schema.set_data_type('teams', 'division', number_allowed=False, strings_allowed='*')
mip_input_schema.set_data_type('teams', 'type', number_allowed=False, strings_allowed=('active', 'dummy'))
mip_input_schema.set_data_type('weeks', 'type', number_allowed=False, strings_allowed=('active', 'dummy'))
mip_input_schema.add_foreign_key('revenue', 'teams', (['home', 'team'], ['away', 'team']))
mip_input_schema.add_foreign_key('revenue', 'weeks', ('week', 'week'))
mip_input_schema.add_parameter('problem ID', 1, nullable=False, must_be_int=True)
mip_input_schema.add_data_row_predicate('revenue', predicate_name='Different Team Check',
                                        predicate=lambda row: row['home'] != row['away'])
