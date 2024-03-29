if [[ "$1" == "" ]]; then
    echo "Error: Missing the dataset name"
    echo "./test_example.sh <dataset name> --<flag (optional)> <content of flag>"
    exit
fi

if [[ "$2" == "--number_fold" ]]; then
    if [[ "$3" == "" ]]; then
        echo "Error: Missing number of fold"
        echo "./test_example.sh --number_fold <number of cross fold>"
        exit
    fi

    for ((i=0; i < $3; i++ ));
    do
        python -m tests.test_seacrowd seacrowd/sea_datasets/$1/$1.py --subset_id "$1_fold$i"
    done
    exit
fi

if [[ "$2" == "--subset_id" ]]; then
    if [[ "$3" == "" ]]; then
        echo "Error: Missing subset_id"
        echo "./test_example.sh --subset_id <subset_id>"
        exit
    fi
    
    python -m tests.test_seacrowd seacrowd/sea_datasets/$1/$1.py --subset_id $3
    exit

elif [[ "$2" == "--data_dir" ]]; then
    if [[ "$3" == "" ]]; then
        echo "Error: Missing data_dir"
        echo "./test_example.sh --data_dir <data_dir>"
        exit
    fi
    
    python -m tests.test_seacrowd seacrowd/sea_datasets/$1/$1.py --data_dir $3
    exit
fi

python -m tests.test_seacrowd seacrowd/sea_datasets/$1/$1.py
