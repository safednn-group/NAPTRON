#!/bin/bash

#run this script from naptron root dir

find "scripts/generate_bboxes" -type f -name "*.sh" ! -name "generate_bboxes_all_methods.sh" -exec bash {} \;

