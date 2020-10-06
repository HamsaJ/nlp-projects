import spacy
from spacy.lang.en import English

nlp = spacy.load(
    "/Users/jama/development/miniconda3/envs/nlp/lib/python3.8/site-packages/en_core_web_lg/en_core_web_lg-2.3.1"
)
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from fastapi import UploadFile
import os
import numpy as np
from .docsim import DocSim

docsim_obj = docsim.DocSim(verbose=True)

from transformers import pipeline


def get_sentiment_analysis(text):
    # Text classification - sentiment analysis
    nlp = pipeline("sentiment-analysis")
    return nlp(text)


async def save_upload_file(upload_file: UploadFile, destination: str, id=-1) -> None:
    try:
        path = Path(destination)
        filename = upload_file.filename
        if not os.path.exists(destination):
            os.makedirs(destination)

        if id != -1:
            filename = f"{id}_{filename}"

        destination = path / filename
        if os.path.exists(destination):
            raise FileExistsError

        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as exception:
        print("Exception: ", exception)
    finally:
        await upload_file.close()


def get_files_in_dir(directory: str) -> None:
    try:
        path = Path(directory)
        return (True, [x for x in path.iterdir() if x.is_file()][0].name)
    except Exception as exception:
        return (False, f"{exception}")


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def handle_upload_file(
    upload_file: UploadFile, handler: Callable[[Path], None]
) -> None:
    tmp_path = save_upload_file_tmp(upload_file)
    try:
        handler(tmp_path)  # Do something with the saved temp file
    finally:
        tmp_path.unlink()  # Delete the temp file


def get_job_and_cvs(directory):
    path = Path(directory)
    job_listing = [x for x in path.iterdir() if x.is_file()][0].name
    cv_list = [x.name for x in list(path.glob("**/*.txt"))]
    return (job_listing, cv_list)


def get_cvranks_for_job(directory):
    path = Path(directory)
    job_listing = [x for x in path.iterdir() if x.is_file()][0]
    job_listing_name = [x for x in path.iterdir() if x.is_file()][0].name

    with job_listing.open() as f:
        job_text = f.read()

    cv_list = [x for x in list(path.glob("cv/*.txt"))]
    titles = [x.name for x in cv_list]
    cv_text_list = list()
    for cv in cv_list:
        with cv.open() as f:
            cv_text = f.read()
            cv_text_list.append(cv_text)

    cosine_sim_score = docsim_obj.similarity_query(job_text, cv_text_list)

    result = list()

    for idx, score in sorted(
        enumerate(cosine_sim_score), reverse=True, key=lambda x: x[1]
    ):
        result.append({"cv": titles[idx], "score": f"{score:0.3f}"})

    return (job_listing_name, result)


def get_cosine_similarity_score(job_list, corpus):
    # Build the term dictionary, TF-idf model
    # The search query must be in the dictionary as well, in case the terms do not overlap with the CVs (we still want similarity)
    dictionary = Dictionary(corpus + [job_list])
    tfidf = TfidfModel(dictionary=dictionary)

    similarity_index = WordEmbeddingSimilarityIndex(glove)
    # Create the term similarity matrix.
    # The nonzero_limit enforces sparsity by limiting the number of non-zero terms in each column.
    # For my application, I got best results by removing the default value of 100
    similarity_matrix = SparseTermSimilarityMatrix(
        similarity_index, dictionary, tfidf
    )  # , nonzero_limit=None)
    # Compute Soft Cosine Measure between the job listing and the CVs.
    # Compute Soft Cosine Measure between the job listing and the CVs.
    query_tf = tfidf[dictionary.doc2bow(job_list)]

    index = SoftCosineSimilarity(
        tfidf[[dictionary.doc2bow(document) for document in corpus]], similarity_matrix
    )

    return index[query_tf]


from gensim.utils import simple_preprocess
import nltk

stopwords = set(nltk.corpus.stopwords.words("english"))


def preprocess(doc):
    # Tokenize, clean up input document string
    doc = sub(r"<img[^<>]+(>|$)", " image_token ", doc)
    doc = sub(r"<[^<>]+(>|$)", " ", doc)
    doc = sub(r"\[img_assist[^]]*?\]", " ", doc)
    doc = sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " url_token ",
        doc,
    )
    return [
        token
        for token in simple_preprocess(doc, min_len=0, max_len=float("inf"))
        if token not in stopwords
    ]
