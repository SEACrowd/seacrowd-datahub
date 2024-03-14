#!/bin/bash

# this simple bash will create a template and making necessary files and copying dataloader template script into dataloader folder dest

if [[ "$1" == "" ]]; then
    echo "Error: Missing the dataset name to be created"
    echo "sh \${YOUR_SEACROWD_ROOT_PATH}/template/initiate_seacrowd_dataloader.sh <dataset name>"
    exit
fi

if [[ "$2" == "" ]]; then
    root_path=./
else
    root_path=$2
fi

(cd $root_path/seacrowd/sea_datasets && mkdir $1 && cd $1 && touch __init__.py)
cp $root_path/templates/template.py $root_path/seacrowd/sea_datasets/$1/$1.py

echo "Initialization is done. Exiting..."
