#!/bin/sh

files="$( find . -path "./venv" -prune -o -type f -name "*.py" | grep -v '^./venv$' )"
by_one="$( printf "%s\n" "$files" | \
    while read -r filename; do \
        echo "$( grep -v '^\s*$' $filename | wc -l ) $( printf "%s" $filename | cut -c3- )"; \
    done)"
echo "$by_one"
printf "total: %s\n" "$( echo "$by_one" | cut -d' ' -f1 | paste -sd+ | bc )"
