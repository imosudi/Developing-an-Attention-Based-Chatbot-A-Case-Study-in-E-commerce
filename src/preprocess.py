import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

def load_and_clean_data(path):
    # Load data with available columns
    data = pd.read_csv(path)
    
    # Use 'inbound' to create intents (True = customer, False = agent)
    data['intent'] = data['inbound'].map({True: 'customer_query', False: 'agent_response'})
    
    # Clean text
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower()))
    
    # Encode labels
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['intent'])
    
    return data, le

def tokenize_inputs(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(list(texts), padding=True, truncation=True, return_tensors='tf')