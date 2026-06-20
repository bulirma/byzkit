#!/bin/sh

jobfiles="$( find ./ -type f -name "job_*.sh" )"
echo "$jobfiles" | while read -r jobfile; do
    sed -i "s/___currentdir___/$( pwd | sed 's/\//\\\//g' )/" "$jobfile"
    sed -i "s/___username___/$( whoami )/" "$jobfile"
done 
