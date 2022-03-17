#!/bin/bash

ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py laneatt {}/config_left.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py laneatt {}/config_left.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py laneatt {}/config_left.json
ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py laneatt {}/config_right.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py laneatt {}/config_right.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py laneatt {}/config_right.json

ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py scnn {}/config_left.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py scnn {}/config_left.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py scnn {}/config_left.json
ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py scnn {}/config_right.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py scnn {}/config_right.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py scnn {}/config_right.json

ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py ultrafast {}/config_left.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py ultrafast {}/config_left.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py ultrafast {}/config_left.json
ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py ultrafast {}/config_right.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py ultrafast {}/config_right.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py ultrafast {}/config_right.json

ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py polylanenet {}/config_left.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py polylanenet {}/config_left.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py polylanenet {}/config_left.json
ls -d tusimple-test/clips/0530/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py polylanenet {}/config_right.json
ls -d tusimple-test/clips/0531/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py polylanenet {}/config_right.json
ls -d tusimple-test/clips/0601/* | sort | head -n 10 | xargs -P4 -L1 -I{} python scripts/run_tusimple_drp_wb.py polylanenet {}/config_right.json
