#!/bin/sh

rsync -urv --files-from=- ./ metacentrum:byzkit/ <<EOF
common
dataset
byztex/template_standalone.tex
EOF

rsync -urv --no-relative --files-from=- ./ metacentrum:byzkit/ <<EOF
metacentrum/prepare_job.sh
metacentrum/job_synth.pbs
notrack/public/dsp1200.zip
EOF
