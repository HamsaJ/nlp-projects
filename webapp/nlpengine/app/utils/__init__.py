import spacy
from spacy.lang.en import English

nlp = spacy.load('/Users/jama/development/miniconda3/envs/nlp/lib/python3.8/site-packages/en_core_web_lg/en_core_web_lg-2.3.1')

from transformers import pipeline
def get_sentiment_analysis(text):
    # Text classification - sentiment analysis
    nlp = pipeline("sentiment-analysis")
    return nlp(text)