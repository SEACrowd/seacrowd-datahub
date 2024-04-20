import re


import re

def parse_text(data):
    """
    Parses a list of text data into English and Lio language lists.

    Args:
        data (list): A list of strings containing text data.

    Returns:
        tuple: A tuple containing two lists: English language data and Lio language data.
    """
    # Clean the data by removing unwanted characters and formatting
    cleaned_data = [re.sub(r'^\ufeff?\d+\t|\b\d+\)\s*', '', re.sub(r'\t+', '\t', text)).replace('\t\n', '').replace('\ufeff', '') for text in data]

    # Split the data into English and Lio lists
    eng_data, lio_data = [], []
    sublist = []

    for item in cleaned_data:
        # Replace tabs with spaces
        item = item.replace('\t', ' ')
        if item != '':
            sublist.append(item)
        else:
            if sublist:
                # Append sublist to result if not empty
                eng_data.append(sublist[-1])
                lio_data.append(sublist[0])
                sublist = []

    # Append the remaining sublist if any
    if sublist:
        eng_data.append(sublist[-1])
        lio_data.append(sublist[0])

    return eng_data, lio_data

def parse_wordlist(data):
    """
    Parses a word list for translation into English, Indonesian, and Nage languages.

    Args:
        data (list): A list of strings containing word list data.

    Returns:
        tuple: A tuple containing three lists: English words, Indonesian words, and Nage words.
    """
    # Clean the data by removing unwanted characters and formatting
    cleaned_data = [re.sub(r'^\ufeff?\d+\t|\b\d+\)\s*', '', re.sub(r'\t+', '\t', text)).replace('\t\n', '').replace('\ufeff', '') for text in data]
    
    # Remove empty elements
    cleaned_data = [elem for elem in cleaned_data if elem != '']
    
    eng_word = []
    nage_word = []
    ind_word = []
    
    for item in cleaned_data:
        split_data = item.split()
        # make sure that we only get [eng, ind, nage] because the actual wordlist have 
        # added 'comment' column and not all the words have their comments
        if len(split_data) == 3: 
            eng_word.append(split_data[0])
            ind_word.append(split_data[1])
            nage_word.append(split_data[2])

    return eng_word, ind_word, nage_word




