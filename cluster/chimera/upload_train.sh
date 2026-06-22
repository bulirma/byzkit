#!/bin/sh

rsync -urv --progress --exclude '__pycache__/' --files-from=- ./ chimera:byzkit/ <<EOF
common
train
EOF

rsync -urv --progress --no-relative --files-from=- ./ chimera:byzkit/ <<EOF
cluster/chimera/prepare_job.sh
cluster/chimera/job_train.sh
notrack/public/sds2k.lmdb
notrack/public/sds12h.lmdb
EOF
