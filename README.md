# harassment-analysis
This repository is for harassment analysis on Safecity data. This work is a research project expanding the work in paper, "SafeCity: Understanding Diverse Forms of Sexual Harassment Personal Stories".
The original data sets for experiment and raw text are available on https://github.com/swkarlekar/safecity. The reports is public available on http://www.maps.safecity.in/reports.
Please contact SafeCity moderators at http://maps.safecity.in/contact for permission before the use of this data. We thank the Safecity moderators for granting the permission of using the data.

The annotation data of key element and story classifications are located in /data directory. The ids ( in each of the train, dev, test file) are corresponding to order of the stories
in each of the train, dev, test dataset published on https://github.com/swkarlekar/safecity. This data is for research purposes only.



The meaning of the integer labels are:
harasser_types:
[
'unspecified' - 0
'relative' - 1
'teacher' - 2
'classmate' - 3 
'friend' - 4
'neighbour' - 5 
'conductor/driver' – 6，7 
'work-related' - 8
'police/guard' – 9， 10
'other' – 11，12
]


location_types:
 ['unspecified' -  0
'street' -  1
'transportation' - 2 
'station/stop' -  3
'house/home/private places (party)' -  4
'shopping place' -  5
'neighbourhood' -  6
'park' -  7
'hotel' -  8
'bush' -  9
'parking lot' -  10
'in school or vicinity' - 11 
'restaurant' - 12
'other' - 13
]

Train and Eval

use trainEval.py to train a model specified by a config file.

model implementations are in models

Single task or joint learning can be selected by setting config.task ( options = ['cls','extraction'])
