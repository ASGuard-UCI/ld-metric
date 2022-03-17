#!/bin/bash

python scripts/replay_comma_more_noupdate.py laneatt &\
python scripts/replay_comma_more_noupdate.py scnn &\
python scripts/replay_comma_more_noupdate.py ultrafast &\
python scripts/replay_comma_more_noupdate.py polylanenet
