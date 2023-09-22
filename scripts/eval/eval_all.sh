#!/bin/bash

#run this script from project root dir

find "scripts/eval" -type f -name "*.sh" ! -name "eval_all.sh" -exec bash {} \;

