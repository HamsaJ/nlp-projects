from flask import Blueprint, request, jsonify, Response
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import textacy
from core.utils import num_of_tokens, num_of_sentences
from core import ner_model, nlp, get_sentiment_analysis
from core.pipelines import RESTCountriesComponent
import os

routesBlueprint = Blueprint("routes", __name__)

ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@routesBlueprint.route("/")
def index():
    return "hello world"


@routesBlueprint.route("/upload", methods=["POST"])
def upload_file():
    # check if the post request has the file part
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["file"]
    print("FILE = ", file)
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == "":
        print(file.filename)
        flash("No selected file")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(os.getcwd() + "/uploads", filename))
        return redirect(url_for("uploaded_file", filename=filename))


@routesBlueprint.route("/ner", methods=["POST"])
def ner():
    data = request.get_json()
    result = data["data"]
    # result = textacy.preprocess_text(result, fix_unicode=True, no_accents=True)
    # result = textacy.preprocess.normalize_whitespace(result)
    # result = textacy.preprocess.fix_bad_unicode(result, normalization='NFC')
    doc = ner_model(result)
    tokens = num_of_tokens(doc, result)
    print(tokens)

    sentences = num_of_sentences(doc)
    print(sentences)

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
    result = jsonify({"ents": ents})
    print(f"Detected ents = {len(ents)}")
    return (result, 200, {"Access-Control-Allow-Origin": "*"})


@routesBlueprint.route("/wordsim", methods=["POST"])
def word_similarity():
    """Word Similary
    Compare two objects, and make a prediction of how similar they are.
    Predicting similarity is useful for building recommendation systems or
    flagging duplicates. For example, you can suggest a user content that's
    similar to what they're currently looking at, or label a support ticket as
    a duplicate if it's very similar to an already existing one.
    """
    data = request.get_json()
    result = data["data"]
    tokens = ner_model(result)

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

    result = jsonify({"wordSimilarity": similarity})
    # result.headers = ['Access-Control-Allow-Origin', '*']
    return (
        result,
        200,
        {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,HEAD,OPTIONS,POST,PUT",
            "Access-Control-Allow-Headers": "Origin, X-Requested-With, Content-Type, Accept, x-client-key, x-client-token, x-client-secret, Authorization",
        },
    )


@routesBlueprint.route("/postag", methods=["POST"])
def POS_tagging():
    """Part-of-speech tagging
    Given a sentence, determine the part of speech for each word. Many words, especially common ones,
    can serve as multiple parts of speech. For example, "book" can be a noun ("the book on the table")
    or verb ("to book a flight"); "set" can be a noun, verb or adjective; and "out" can be any of at
    least five different parts of speech. Some languages have more such ambiguity than others.[dubious – discuss]
    Languages with little inflectional morphology, such as English, are particularly prone to such ambiguity.
    Chinese is prone to such ambiguity because it is a tonal language during verbalization.
    Such inflection is not readily conveyed via the entities employed within the orthography to convey intended meaning.
    """
    data = request.get_json()
    result = data["data"]
    doc = ner_model(result)
    pos_tag = [
        ({"text": token.text, "partOfSpeach": token.pos_, "tag": token.tag_})
        for token in doc
    ]
    result = jsonify({"posTags": pos_tag})
    return result


@routesBlueprint.route("/dependency", methods=["POST"])
def dependecy_parsing():
    """Dependecy Parsing
    The parser also powers the sentence boundary detection, and lets you iterate over
    base noun phrases, or "chunks". You can check whether a Doc  object has been parsed
    with the doc.is_parsed attribute, which returns a boolean value. If this attribute
    is False, the default sentence iterator will raise an exception.
    """
    data = request.get_json()
    result = data["data"]
    result = textacy.preprocess_text(result, fix_unicode=True, no_accents=True)
    result = textacy.preprocess.fix_bad_unicode(result, normalization="NFC")
    doc = ner_model(result)

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
    result = jsonify({"dependency": dependency})
    return result


@routesBlueprint.route("/sentan", methods=["POST"])
def sentiment_analysis():
    """Dependecy Parsing
    The parser also powers the sentence boundary detection, and lets you iterate over
    base noun phrases, or "chunks". You can check whether a Doc  object has been parsed
    with the doc.is_parsed attribute, which returns a boolean value. If this attribute
    is False, the default sentence iterator will raise an exception.
    """
    data = request.get_json()
    result = get_sentiment_analysis(data["data"])
    result = jsonify({"sentAnalysis": result})
    return result


@routesBlueprint.route("/country", methods=["POST"])
def country_detector():
    """Country detector"""
    data = request.get_json()
    result = data["data"]
    doc = nlp(result)
    print("Pipeline", nlp.pipe_names)  # pipeline contains component name
    print("Doc has countries", doc._.has_country)  # Doc contains countries
    additional_info = []
    for token in doc:
        if token._.is_country:
            info = (
                token.text,
                token._.country_capital,
                token._.country_latlng,
                token._.country_flag,
            )
            additional_info.append(info)

    result = jsonify(
        {
            "Entities": [(e.text, e.label_) for e in doc.ents],
            "additional_info": additional_info,
        }
    )
    return result
