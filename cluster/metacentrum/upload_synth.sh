#!/bin/sh

rsync -urv --progress --exclude '__pycache__/' --files-from=- ./ metacentrum:byzkit/ <<EOF
common
dataset
byztex/standalone_neumes.txt
EOF

rsync -urv --progress --no-relative --files-from=- ./ metacentrum:byzkit/ <<EOF
cluster/metacentrum/prepare_job.sh
cluster/metacentrum/job_synth.pbs
notrack/public/dsp2k.zip
EOF
