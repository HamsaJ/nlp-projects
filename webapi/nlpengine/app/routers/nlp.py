from fastapi import APIRouter
from textacy import preprocessing
from ..utils import get_sentiment_analysis, nlp, DocumentReader

reader = DocumentReader()
router = APIRouter()

from pydantic import BaseModel


class Data(BaseModel):
    text: str


@router.post("/nlp/ner", tags=["nlp"])
async def nlp_ner(data: Data):
    doc = nlp(data.text)
    ents = [
        (
            {
                "label": ent.label_,
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )
        for ent in doc.ents
    ]

    return {"ents": ents}


@router.post("/nlp/wordsim", tags=["nlp"])
async def nlp_wordsim(data: Data):
    """Word Similary
    Compare two objects, and make a prediction of how similar they are.
    Predicting similarity is useful for building recommendation systems or
    flagging duplicates. For example, you can suggest a user content that's
    similar to what they're currently looking at, or label a support ticket as
    a duplicate if it's very similar to an already existing one.
    """

    tokens = nlp(data.text)

    similarity = [
        (
            {
                "token1": token1.text,
                "token2": token2.text,
                "similarity": str(token1.similarity(token2)),
            }
        )
        for token2 in tokens
        for token1 in tokens
    ]

    result = {"wordSimilarity": similarity}
    # result.headers = ['Access-Control-Allow-Origin', '*']
    return result


@router.post("/nlp/postag", tags=["nlp"])
async def nlp_postag(data: Data):
    """Part-of-speech tagging
    Given a sentence, determine the part of speech for each word. Many words, especially common ones,
    can serve as multiple parts of speech. For example, "book" can be a noun ("the book on the table")
    or verb ("to book a flight"); "set" can be a noun, verb or adjective; and "out" can be any of at
    least five different parts of speech. Some languages have more such ambiguity than others.[dubious – discuss]
    Languages with little inflectional morphology, such as English, are particularly prone to such ambiguity.
    Chinese is prone to such ambiguity because it is a tonal language during verbalization.
    Such inflection is not readily conveyed via the entities employed within the orthography to convey intended meaning.
    """

    doc = nlp(result)
    pos_tag = [
        ({"text": token.text, "partOfSpeach": token.pos_, "tag": token.tag_})
        for token in doc
    ]
    result = {"posTags": pos_tag}
    return result


@router.post("/nlp/sentanalysis", tags=["nlp"])
async def nlp_sentiment_analysis(data: Data):
    result = get_sentiment_analysis(data.text)
    return {"sentanalysis": result}


@router.post("/nlp/dependency", tags=["nlp"])
async def nlp_dependency(data: Data):
    """Dependecy Parsing
    The parser also powers the sentence boundary detection, and lets you iterate over
    base noun phrases, or "chunks". You can check whether a Doc  object has been parsed
    with the doc.is_parsed attribute, which returns a boolean value. If this attribute
    is False, the default sentence iterator will raise an exception.
    """

    result = preprocessing.normalize_whitespace(data.text)
    result = preprocessing.normalize_unicode(result, form="NFC")
    doc = nlp(result)

    dependency = [
        (
            {
                "text": token.text,
                "dependency": token.dep_,
                "tokenHead": token.head.text,
                "tokenHeadPartOfSpeach": token.head.pos_,
                "children": [str(child) for child in token.children],
            }
        )
        for token in doc
    ]
    result = {"dependency": dependency}
    return result


@router.post("/nlp/questionanswer", tags=["nlp"])
async def nlp_question_and_answer(question: str, context: str):

    reader.tokenize(question, context)

    result = {"answer": reader.get_answer()}
    return result