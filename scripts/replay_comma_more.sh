#!/bin/bash

python scripts/replay_comma_more.py laneatt &\
python scripts/replay_comma_more.py scnn &\
python scripts/replay_comma_more.py ultrafast  &\
python scripts/ replay_comma_more.py polylanenet
