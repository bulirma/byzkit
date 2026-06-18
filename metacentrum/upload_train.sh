#!/bin/sh

rsync -urv --progress --files-from=- ./ metacentrum:byzkit/ <<EOF
common
train
EOF

rsync -urv --progress --no-relative --files-from=- ./ metacentrum:byzkit/ <<EOF
metacentrum/prepare_job.sh
metacentrum/job_train.pbs
notrack/public/sds2k.lmdb
EOF
