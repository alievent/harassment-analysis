UNK = "<unk>"
PAD = "<pad>"
POS_UNK = "<pos_unk>"
POS_PAD = "<pos_pad>"
delimiters = ['\n','.','?','!']
tagMap ={ "None" : 0, "harasser" : 1, "time": 2, "location": 3, "trigger words" : 4 }
id2Tags =["None", "Harasser", "Time", "Locati:on", "Trigger"]


harasser_age_map = ['unspecified','young','adult']
harasser_num_map = ['unspecified','one', 'multiple']

harasser_type_map = ['unspecified','relative','teacher','classmate','friend','neighbour','conductor/driver','work-related','police/guard','other']

# "friend" : directly or indirectly related to the victim as a friend
# "police/guard" : someone are supposed to protecting people, usually their profession giving the impression of authority
# 'other': specified professions but too few
# "conductor/driver" : personal that conduct/operate an vehicle or transportation
# "neighbour" : people from neigbourhood
time_of_day_map = ['unspecified', 'day', 'night'] # 5am to 6pm day
location_type_map = ['unspecified','street','transportation','station/stop','house/home/private places (party)','shopping place','neighbourhood','park','hotel','bush','parking lot','in school or vicinity','restaurant','other']

