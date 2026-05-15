#!/bin/sh

jobfiles="$( find ./ -type f -name "*.pbs" )"
echo "$jobfiles" | while read -r jobfile; do
    sed -i "s/___currentdir___/$( pwd | sed 's/\//\\\//g' )/" "$jobfile"
done 
