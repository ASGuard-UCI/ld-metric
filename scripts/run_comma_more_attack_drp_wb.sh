echo "" > nums.txt
#!/bin/bash

for offset in {1..100} ; do
    echo ${offset} >> nums.txt
done

cat nums.txt | xargs -P2 -L1 -I{} python scripts/run_comma_more_attack_drp_wb.py laneatt {}
cat nums.txt | xargs -P2 -L1 -I{} python scripts/run_comma_more_attack_drp_wb.py scnn {}
cat nums.txt | xargs -P2 -L1 -I{} python scripts/run_comma_more_attack_drp_wb.py ultrafast {}
cat nums.txt | xargs -P2 -L1 -I{} python scripts/run_comma_more_attack_drp_wb.py polylanenet {}