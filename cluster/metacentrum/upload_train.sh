#!/bin/sh

rsync -urv --progress --files-from=- ./ metacentrum:byzkit/ <<EOF
common
train
EOF

rsync -urv --progress --no-relative --files-from=- ./ metacentrum:byzkit/ <<EOF
cluster/metacentrum/prepare_job.sh
cluster/metacentrum/job_train.pbs
notrack/public/sds2k.lmdb
EOF
