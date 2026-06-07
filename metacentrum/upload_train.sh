#!/bin/sh

rsync -urv --files-from=- ./ metacentrum:byzkit/ <<EOF
common
train
EOF

rsync -urv --no-relative --files-from=- ./ metacentrum:byzkit/ <<EOF
metacentrum/prepare_job.sh
metacentrum/job_train.pbs
notrack/public/sds1200.lmdb
EOF
