# harassment-analysis
This repositary is for harassment analysis on safeCity data. 

story_and_category.csv file
-	This file contains all the stories up to date and their corresponding categories regarding the harassment types

key_info_and_classes_part.csv
-	Currently only contains part of the labeled online stories. 
-	More stories will be added after they are cleaned, e.g. sensitive data being handled properly.

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
- The "extraction_file" refer to path to key element annotation files (.json).
- The "classification_file" refer to class label file (.csv)