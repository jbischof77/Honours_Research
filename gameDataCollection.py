from riotwatcher import LolWatcher
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

api_key = "RGAPI-16e158f4-bcd4-4e05-81d9-68940c75400b"
watcher = LolWatcher(api_key)
region_v4 = "NA1"
region_v5 = "AMERICAS"

player = watcher.summoner.by_name(region_v4, 'Doublelift')

matches = watcher.match.matchlist_by_puuid(region_v5, player['puuid'], count = 100)                                              #get the last 20 games played of the player specified in player

timeline = []
i = 0

for matchId in matches: 
    ##loop to go through all 20 matches and collect the relevant data##
    match_timeline_detail = watcher.match.timeline_by_match(region_v5, matchId)
    match_detail = watcher.match.by_id(region_v5, matchId)

    individualTimelines = []
    participants = []

    for row in match_timeline_detail['info']['participants']:
        ##data correlating players to their champions##
        participants_row = {}
        participants_row['puuid'] = row['puuid']
        participants_row['participantId'] = row['participantId']
        for row1 in match_detail['info']['participants']:
            if row['puuid'] == row1['puuid']:
                participants_row['champion'] = row1['championName']
        participants.append(participants_row)

    for frame in match_timeline_detail['info']['frames']:
        for row in frame['events']:
            ##data collection for events from matches begins here##
            if(row['type'] == "CHAMPION_KILL"
            or row['type'] == 'CHAMPION_SPECIAL_KILL'
            or row['type'] == 'TURRET_PLATE_DESTROYED'
            or row['type'] == 'ELITE_MONSTER_KILL'
            or row['type'] == 'BUILDING_KILL'):                                                                             #the event types that have been chosen to be included

                timeline_row = {}
                timeline_row['timestamp'] = row['timestamp']                                                                #timestamp of the event
                timeline_row['type'] = row['type']                                                                          #event type
                timeline_row['killerId'] = row['killerId']                                                                  #id of the killer

                if row['killerId'] <= 5 and row['killerId'] > 0:                                                            #determine the team of the killer
                    timeline_row['killerTeam'] = "Blue"
                elif row['killerId'] > 5 and row['killerId'] <= 10:
                    timeline_row['killerTeam'] = "Red"
                else:
                    timeline_row['killerTeam'] = "No team"

                timeline_row['x'] = row['position']['x']                                                                    #where did the event occur on the map
                timeline_row['y'] = row['position']['y']

                if row['type'] == 'CHAMPION_KILL':

                    for victim in participants:
                        if row['victimId'] == victim['participantId']:
                            timeline_row['killType'] = "{0}_KILL".format(victim['participantId'])                        #gives the name of the champion that was killed

                elif row['type'] == 'CHAMPION_SPECIAL_KILL':

                    timeline_row['killType'] = row['killType']                                                              #the type of special kill that occured

                elif row['type'] == 'TURRET_PLATE_DESTROYED':

                    if row['teamId'] == 200:                                                                                #determines the team who destroyeda turret plate
                        timeline_row['killerTeam'] = "Blue"
                    elif row['teamId'] == 100:
                        timeline_row['killerTeam'] = "Red"
                    else:
                        timeline_row['killerTeam'] = "No team"

                    timeline_row['killType'] = "TURRET_PLATE"

                elif row['type'] == 'ELITE_MONSTER_KILL':

                    timeline_row['killType'] = row['monsterType']                                                           #gives the type of monster that was killed

                elif row['type'] == 'BUILDING_KILL':

                    timeline_row['killType'] = row['buildingType']                                                          #gives the type of 

                    if row['teamId'] == 200:                                                                                #determines the team that destroyed the building
                        timeline_row['killerTeam'] = "Blue"
                    elif row['teamId'] == 100:
                        timeline_row['killerTeam'] = "Red"
                    else:
                        timeline_row['killerTeam'] = "No team"

                timeline.append(timeline_row)                                                                               #creates a list of dictionaries of the events of a match
                individualTimelines.append(timeline_row)
    
    # df = pd.DataFrame(individualTimelines)
    # df.to_csv(r'game{0}_data.csv'.format(i), index = False)

    i += 1
        
df = pd.DataFrame(timeline)                                                                                                 #makes a data frame from the list of dictionaries
df.to_csv(r'timeline_data_large.csv', index = False)

# print(df)

img = plt.imread("Images\Summoner's_Rift_Minimap.jpg")
fig, ax =  plt.subplots(figsize = (10, 10))

sns.scatterplot(x = 'x', y = 'y', hue = 'killerTeam', data = df, ax = ax, style = 'type')

ax.imshow(img, extent = [0, 14800, 0, 14900], aspect = 'auto')
plt.show()