def load_data(filename:str) -> str:
    with open(filename, encoding='utf8', mode='r') as f:
        data = f.read()
    return data

def save_file(filename:str, path:str) -> bool:
    try:  
        with open('data/10k/apple-20191031.txt', encoding='utf8', mode='w+') as f:
            f.write(raw_10k)
            return True
    except Exception as e:
        print(e)
        return False
    
def fetch_from_url(url:str) -> str:
    try:        
        import requests
        res = requests.get(url)
        return res.text
    except Exception as e:
        print(e)
        return false
    
def parsed_10k_data(raw_10k:str)-> dict:
    import re

    # Regex to find <DOCUMENT> tags
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag prceeding any characters, terminating at new line
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    # Create 3 lists with the span idices for each regex

    ### There are many <Document> Tags in this text file, each as specific exhibit like 10-K, EX-10.17 etc
    ### First filter will give us document tag start <end> and document tag end's <start> 
    ### We will use this to later grab content in between these tags
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]

    ### Type filter is interesting, it looks for <TYPE> with Not flag as new line, ie terminare there, with + sign
    ### to look for any char afterwards until new line \n. This will give us <TYPE> followed Section Name like '10-K'
    ### Once we have have this, it returns String Array, below line will with find content after <TYPE> ie, '10-K' 
    ### as section names
    doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]
    document = {}

    # Create a loop to go through each section type and save only the 10-K section in the dictionary
    for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
        if doc_type == '10-K':
            document[doc_type] = raw_10k[doc_start:doc_end]
    return document

def cleaned_dataframe(document:dict):
    import re
    
    # import pandas
    import pandas as pd
    
    # Write the regex
    regex = re.compile(r'(>Item(\s|&#160;|&nbsp;)(1A|1B|7A|7|8)\.{0,1})|(ITEM\s(1A|1B|7A|7|8))')

    # Use finditer to math the regex
    matches = regex.finditer(document['10-K'])
    
    # Matches
    matches = regex.finditer(document['10-K'])

    # Create the dataframe
    test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])

    test_df.columns = ['item', 'start', 'end']
    test_df['item'] = test_df.item.str.lower()
    
    # Get rid of unnesesary charcters from the dataframe
    test_df.replace('&#160;',' ',regex=True,inplace=True)
    test_df.replace('&nbsp;',' ',regex=True,inplace=True)
    test_df.replace(' ','',regex=True,inplace=True)
    test_df.replace('\.','',regex=True,inplace=True)
    test_df.replace('>','',regex=True,inplace=True)
    
    # Drop duplicates
    pos_dat = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='last')
    # Set item as the dataframe index
    pos_dat.set_index('item', inplace=True)
    
    return pos_dat