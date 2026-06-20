#!/bin/sh

rsync -urv --progress --files-from=- ./ chimera:byzkit/ <<EOF
common
train
EOF

rsync -urv --progress --no-relative --files-from=- ./ chimera:byzkit/ <<EOF
cluster/chimera/prepare_job.sh
cluster/chimera/job_train.sh
notrack/public/sds12h.lmdb
EOF
