#!/bin/sh

rsync -urv --progress "chimera:byzkit/$1" ./tmp/
