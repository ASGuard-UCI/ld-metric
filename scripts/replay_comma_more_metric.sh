#!/bin/bash

python scripts/replay_comma_more_metric.py laneatt &\
python scripts/replay_comma_more_metric.py scnn &\
python scripts/replay_comma_more_metric.py ultrafast &\
python scripts/replay_comma_more_metric.py polylanenet

