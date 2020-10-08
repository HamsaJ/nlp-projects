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


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

from collections import OrderedDict


class DocumentReader:
    def __init__(
        self,
        pretrained_model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad",
    ):
        self.READER_PATH = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def tokenize(self, question, text):
        self.inputs = self.tokenizer.encode_plus(
            question, text, add_special_tokens=True, return_tensors="pt"
        )
        self.input_ids = self.inputs["input_ids"].tolist()[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

    def chunkify(self):
        """
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model.

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = self.inputs["token_type_ids"].lt(1)
        qt = torch.masked_select(self.inputs["input_ids"], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1  # the "-1" accounts for
        print(f"Each chunk will contain {chunk_size - 2}")
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k, v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)

            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks) - 1:
                    if k == "input_ids":
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self):
        if self.chunked:
            answer = ""
            for k, chunk in self.inputs.items():
                answer_start_scores, answer_end_scores = self.model(**chunk)

                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                ans = self.convert_ids_to_string(
                    chunk["input_ids"][0][answer_start:answer_end]
                )
                if ans != "[CLS]":
                    answer += ans + " / "
            return answer.split("/  / [SEP]")[0]
        else:
            answer_start_scores, answer_end_scores = self.model(**self.inputs)

            answer_start = torch.argmax(
                answer_start_scores
            )  # get the most likely beginning of answer with the argmax of the score
            answer_end = (
                torch.argmax(answer_end_scores) + 1
            )  # get the most likely end of answer with the argmax of the score

            return self.convert_ids_to_string(
                self.inputs["input_ids"][0][answer_start:answer_end]
            )

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids)
        )


"""
:mod:`textacy.preprocessing.normalize`: Normalize aspects of raw text that may vary
in problematic ways.
"""
import re
import unicodedata

from .resources import (
    QUOTE_TRANSLATION_TABLE,
    RE_HYPHENATED_WORD,
    RE_LINEBREAK,
    RE_NONBREAKING_SPACE,
    RE_ZWSP,
)


def normalize_hyphenated_words(text: str) -> str:
    """
    Normalize words in ``text`` that have been split across lines by a hyphen
    for visual consistency (aka hyphenated) by joining the pieces back together,
    sans hyphen and whitespace.
    """
    return RE_HYPHENATED_WORD.sub(r"\1\2", text)


def normalize_quotation_marks(text: str) -> str:
    """
    Normalize all "fancy" single- and double-quotation marks in ``text``
    to just the basic ASCII equivalents. Note that this will also normalize fancy
    apostrophes, which are typically represented as single quotation marks.
    """
    return text.translate(QUOTE_TRANSLATION_TABLE)


def normalize_repeating_chars(text: str, *, chars: str, maxn: int = 1) -> str:
    """
    Normalize repeating characters in ``text`` by truncating their number of consecutive
    repetitions to ``maxn``.
    Args:
        text
        chars: One or more characters whose consecutive repetitions are to be normalized,
            e.g. "." or "?!".
        maxn: Maximum number of consecutive repetitions of ``chars`` to which
            longer repetitions will be truncated.
    Returns:
        str
    """
    return re.sub(r"({}){{{},}}".format(re.escape(chars), maxn + 1), chars * maxn, text)


def normalize_unicode(text: str, *, form: str = "NFC") -> str:
    """
    Normalize unicode characters in ``text`` into canonical forms.
    Args:
        text
        form ({"NFC", "NFD", "NFKC", "NFKD"}): Form of normalization applied to
            unicode characters. For example, an "e" with accute accent "´" can be
            written as "e´" (canonical decomposition, "NFD") or "é" (canonical
            composition, "NFC"). Unicode can be normalized to NFC form
            without any change in meaning, so it's usually a safe bet. If "NFKC",
            additional normalizations are applied that can change characters' meanings,
            e.g. ellipsis characters are replaced with three periods.
    See Also:
        https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
    """
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """
    Replace all contiguous zero-width spaces with an empty string, line-breaking spaces
    with a single newline, and non-breaking spaces with a single space, then
    strip any leading/trailing whitespace.
    """
    text = RE_ZWSP.sub("", text)
    text = RE_LINEBREAK.sub(r"\n", text)
    text = RE_NONBREAKING_SPACE.sub(" ", text)
    return text.strip()