import pandas as pd

input=pd.read_excel('rawdata.xlsx',sheet_name=0)
output=pd.read_excel('rawdata.xlsx',sheet_name=1)
hold=output
placed_counts = input[input['activity'] == 'placed'].groupby('date').size()
picked_counts = input[input['activity'] == 'picked'].groupby('date').size()

output['pick_activities'] = output['date'].map(picked_counts)
output['place_activities'] = output['date'].map(placed_counts)


input['date'] = pd.to_datetime(input['date'])

input['time'] = input['time'].astype(str)

input['datetime'] = pd.to_datetime(input['date'].astype(str) + ' ' + input['time'])

input = input.sort_values(by='datetime')

total_duration = input.groupby(['date', 'position'])['position'].count().unstack(fill_value=0).reset_index()
total_duration['inside_total'] = total_duration['Inside'] + total_duration['inside']

output['inside_duration'] = total_duration['inside_total'].values
output['outside_duration'] = total_duration['outside'].values


print("done")