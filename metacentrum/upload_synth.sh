#!/bin/sh

rsync -urv --files-from=- ./ metacentrum:byzkit/ <<EOF
metacentrum/prepare_job.sh
metacentrum/job_synth.pbs
common
dataset
byzkit.py
byztex/template_standalone.tex
notrack/public/dsp1200.zip
EOF
