    
labels_dict = {
    "aaron_staton": 3,
    "adam_copeland": 17,
    "amitabh_bachchan": 101,
    "andrew_rannells": 119,
    "ben_whishaw": 202,
    "chris_riggi": 366,
    "connie_nielsen": 416,
    "elaine_hendrix": 632,
    "eli_roth": 636,
    "jeannie_mai": 1007,
    "jennifer_ferrin": 1040,
    "john_mahoney": 1164,
    "josh_lucas": 1224,
    "katherine_jenkins": 1296,
    "kathryn_mccormick": 1309,
    "kelly_rowan": 1363,
    "kiersten_warren": 1389,
    "lana_wachowski": 1427,
    "lauren_bowles": 1440,
    "lennie_james": 1464,
    "marshall_allman": 1630,
    "michael_gambon": 1737,
    "moran_atias": 1807,
    "ralf_little": 2019,
    "salli_richardson-whitfield": 2163,
    "tommy_morrison": 2485,
    "tracy_morgan": 2494,
    "virginia_williams": 2543,
    "wagner_moura": 2546,
    "zac_efron": 2603
}

labels = labels_dict.values()

names = [line.rstrip('\n') for line in open('data/names.txt')]

for label in labels:
    print(names[label])
