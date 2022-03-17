#!/bin/bash

ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py laneatt {}/config_left.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py laneatt {}/config_left.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py laneatt {}/config_left.json

ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py scnn {}/config_left.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py scnn {}/config_left.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py scnn {}/config_left.json

ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py ultrafast {}/config_left.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py ultrafast {}/config_left.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py ultrafast {}/config_left.json

ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py polylanenet {}/config_left.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py polylanenet {}/config_left.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/replay_tusimple_benign.py polylanenet {}/config_left.json
