#!/bin/bash

# Example script to download, unpack and prepare a dataset

#destination PARENT path for kitty tiny dataset
destination_path=$1

echo "chosen destination path: ${destination_path} "

cd ${destination_path}

/bin/bash ../scripts/gdrive_download2.sh 1n9C4CiBURMSCZy2LStBQTzR17rD_a67e voc_07_12.zip
unzip voc_07_12.zip
rm voc_07_12.zip