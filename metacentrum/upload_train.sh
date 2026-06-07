#!/bin/sh

rsync -urv --files-from=- ./ metacentrum:byzkit/ <<EOF
metacentrum/prepare_job.sh
metacentrum/job_train.pbs
metacentrum/job_dummy_train.pbs
common
train
byzkit.py
byztex/template_standalone.tex
notrack/public/sd1200.lmdb
EOF
