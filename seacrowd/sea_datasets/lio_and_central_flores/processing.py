import re

def parse_text(data):
    cleaned_data = [re.sub(r'^\ufeff?\d+\t|\b\d+\)\s*', '', re.sub(r'\t+', '\t', text)).replace('\t\n', '').replace('\ufeff', '') for text in data]

    result, sublist = [], []
    for item in cleaned_data:
        item = item.replace('\t', ' ')
        if item != '':
            sublist.append(item)
        else:
            if sublist:
                result.append(sublist)
                sublist = []

    if sublist:
        result.append(sublist)

    lio_data = []
    eng_data = []

    for i in result:
      if len(i) == 2:
        lio_data.append(i[0])
        eng_data.append(i[1])
      elif len(i) == 3:
        lio_data.append(i[0])
        eng_data.append(i[2])
      elif len(i) == 4:
        lio_data.append(i[1])
        eng_data.append(i[3])
        
    return eng_data, lio_data

def parse_wordlist(data):
    """
    untuk Nage
    """
    cleaned_data = [re.sub(r'^\ufeff?\d+\t|\b\d+\)\s*', '', re.sub(r'\t+', '\t', text)).replace('\t\n', '').replace('\ufeff', '') for text in data]
    cleaned_data = [elem for elem in cleaned_data if elem != '']
    
    eng_word = []
    nage_word = []
    ind_word = []
    for item in cleaned_data:
        split_data = item.split()
        if len(split_data) == 3:
            eng_word.append(split_data[0])
            nage_word.append(split_data[2])
            ind_word.append(split_data[1])
            
    return eng_word, ind_word, nage_word



