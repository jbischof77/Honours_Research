import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def TowerPlacements(df):
    # tower_df = df.loc[(df['killType'] == 'TOWER_BUILDING')]

    # x = tower_df['x'].unique()
    # y = tower_df['y'].unique()

    for i in df.index:# for i, j in zip(x, y):
        if(df['killType'][i] == "TOWER_BUILDING"):
            if((df['x'][i] < 3000 and df['y'][i] < 3000) or (df['x'][i] > 12000 and df['y'][i] > 12000)):
                # print("Nexus Tower")
                # print(df['x'][i], df['y'][i])
                df['killType'][i] = "NEXUS_TOWER"
            elif((df['x'][i] < 2000 and df['y'][i] > 4000) or (df['x'][i] < 12000 and df['y'][i] > 12000)):
                # print("Top Tower")
                # print(df['x'][i], df['y'][i])
                df['killType'][i] = "TOP_TOWER"
            elif((df['x'][i] > 4000 and df['y'][i] < 2000) or (df['x'][i] > 12000 and df['y'][i] > 2000)):
                # print("Bottom Tower")
                # print(df['x'][i], df['y'][i])
                df['killType'][i] = "BOT_TOWER"
            else:
                # print("Middle Tower")
                # print(df['x'][i], df['y'][i])
                df['killType'][i] = "MID_TOWER"

    return df

def MapEventsForRoles(palette, pos, team, train_df):
    for i in range(1, 6):
        kill_df = pd.concat([train_df.loc[(train_df['killerId'] == i)], train_df.loc[(train_df['killerId'] == i + 5)]])

        img = plt.imread("Images\Summoner's_Rift_Minimap.jpg")
        fig, ax =  plt.subplots(figsize = (10, 10))

        img_path = "Images\\{}_Mapped.png".format(pos[(i - 1)])

        sns.scatterplot(x = 'x', y = 'y', hue = 'type', data = kill_df, ax = ax, palette = palette)

        ax.imshow(img, extent = [0, 14800, 0, 14900], aspect = 'auto')
        plt.legend(bbox_to_anchor = (1, 1), loc = 2, borderaxespad = 0.)
        plt.show()
        # plt.savefig(img_path, bbox_inches = 'tight')
        # plt.close()

def GameTimelineForRoles(kills, palette, pos, train_df):
    for i in range(1, 6):
        kill_df = pd.concat([train_df.loc[(train_df['killerId'] == i)], train_df.loc[(train_df['killerId'] == i + 5)]])
        # kill_df["killerId"] = kill_df['killerId'].replace([i, i + 5], pos[i - 1])
        # kill_df["killType"] = kill_df['killType'].replace(kills, "CHAMPION_KILL")
        kill_df["killType"] = kill_df['killType'].replace(["1_KILL", "6_KILL"], "TOP_KILL")
        kill_df["killType"] = kill_df['killType'].replace(["2_KILL", "7_KILL"], "JUNG_KILL")
        kill_df["killType"] = kill_df['killType'].replace(["3_KILL", "8_KILL"], "MID_KILL")
        kill_df["killType"] = kill_df['killType'].replace(["4_KILL", "9_KILL"], "ADC_KILL")
        kill_df["killType"] = kill_df['killType'].replace(["5_KILL", "10_KILL"], "SUPP_KILL")
        kill_df["killType"] = kill_df['killType'].replace(["BARON_NASHOR", "RIFTHERALD"], "RIFT_MONSTER")
        # print(kill_df)
        # quit()

        img_path = "Images\\timeline_{}_specific_champion_and_tower_kills.png".format(pos[i - 1])
        title = "Timeline {}".format(pos[i - 1])

        fig, ax = plt.subplots(1, 1)

        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(visible = True, which = 'major', color = 'black', linewidth = 0.5)
        ax.grid(visible = True, which = 'minor', color = 'black', linewidth = 0.5)

        sns.stripplot(x = 'timestamp', y = 'killType', order = ['TOP_KILL', 'TOP_TOWER', 'RIFT_MONSTER', 'MID_KILL', 'MID_TOWER', 'ADC_KILL', 'SUPP_KILL', 'BOT_TOWER', 'DRAGON', 'JUNG_KILL', 'INHIBITOR_BUILDING', 'NEXUS_TOWER'], data = kill_df, hue = 'type', palette = palette).set(title = title) #order = ['CHAMPION_KILL', 'DRAGON', 'RIFT_MONSTER', 'TOWER_BUILDING', 'INHIBITOR_BUILDING'],

        plt.legend(bbox_to_anchor = (1, 1), loc = 2, borderaxespad = 0.)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()
        # plt.savefig(img_path, bbox_inches = 'tight')
        # plt.close()

def fullGameTimePlot(palette):
    for i in range(0, 20):
        csv_path = "CSV\\game{}_data.csv".format(i)
        df = pd.read_csv(csv_path)

        plt.ylim(-1, 11)

        sns.scatterplot(x = 'timestamp', y = 'killerId', data = df, hue = 'type', palette = palette)
        
        plt.legend(bbox_to_anchor = (1, 1), loc = 2, borderaxespad = 0.)
        img_path = "Images\\game{}_timeline.png".format(i)
        plt.show()
        # plt.savefig(img_path, bbox_inches = 'tight')
        # plt.close()

if __name__ == "__main__":
    # train_df = pd.read_csv("CSV\\timeline_data_mini.csv")
    train_df = pd.read_csv("CSV\\timeline_data_large.csv")
    team = ["Blue", "Red"]
    pos = ["Top", "Jung", "Mid", "ADC", "Supp"]
    palette = {"CHAMPION_KILL" : "C0", "CHAMPION_SPECIAL_KILL" : "C1", "TURRET_PLATE_DESTROYED" : "C2", "ELITE_MONSTER_KILL" : "C3", "BUILDING_KILL" : "C4"}
    kills = ["1_KILL", 
            "2_KILL", 
            "3_KILL", 
            "4_KILL", 
            "5_KILL", 
            "6_KILL", 
            "7_KILL", 
            "8_KILL", 
            "9_KILL", 
            "10_KILL",
            "KILL_FIRST_BLOOD", 
            "KILL_MULTI",
            "KILL_ACE"]
    
    MapEventsForRoles(palette, pos, team, train_df)
    # fullGameTimePlot(palette)
    new_train_df = TowerPlacements(train_df)
    GameTimelineForRoles(kills, palette, pos, new_train_df)