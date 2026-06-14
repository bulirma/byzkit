#!/bin/sh

rsync -urv --files-from=- ./ metacentrum:byzkit/ <<EOF
common
dataset
byztex/standalone_neumes.txt
EOF

rsync -urv --no-relative --files-from=- ./ metacentrum:byzkit/ <<EOF
metacentrum/prepare_job.sh
metacentrum/job_synth.pbs
notrack/public/dsp2k.zip
EOF
