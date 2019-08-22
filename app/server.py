# flask_app/server.py​
import datetime
import re

from flask import Flask, request, jsonify, render_template, session, url_for, redirect
from flask_dropzone import Dropzone
import time
from urllib.parse import unquote
import os
import uuid
import secrets

from run_squad import initialize, evaluate
from squad_generator import convert_text_input_to_squad, \
    convert_file_input_to_squad, convert_context_and_questions_to_squad
from settings import *
import requests

# args, model, tokenizer = None, None, None
args, model, tokenizer = initialize()

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'text'
app.config.update(
    SECRET_KEY=secrets.token_urlsafe(32),
    SESSION_COOKIE_NAME='InteractiveTransformer-WebSession'
)

dropzone = Dropzone(app)

def delay_func(func):
    def inner(*args, **kwargs):
        returned_value = func(*args, **kwargs)
        time.sleep(0)
        return returned_value
    return inner

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def process_input():
    if request.files:
        if "file_urls" not in session:
            session['file_urls'] = []
            # list to hold our uploaded image urls
        file_urls = session['file_urls']

        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            app.logger.info("file upload {}".format(file.filename))
            os.makedirs("./uploads", exist_ok=True)
            filepath = os.path.join('./uploads', secrets.token_urlsafe(8))
            file.save(filepath)
            file_urls.append(filepath)
        return "upload"
    else:
        input = request.form["textbox"]
        try:
            return predict_from_text_squad(input)
        except AssertionError:
            return index()

@app.route("/_random_page")
def random_page():
    # r = wikipedia.random(1)
    # try:
    #     res = wikipedia.page(r)
    #     res_title = res.title
    #     res_sum = res.summary
    # except wikipedia.exceptions.DisambiguationError as e:
    #     return random_page()
    # return jsonify(context='\n'.join([res_title, res_sum]))
    if proxyDict:
        r = requests.get("https://en.wikipedia.org/api/rest_v1/page/random/summary", proxies=proxyDict)
    else:
        r = requests.get("https://en.wikipedia.org/api/rest_v1/page/random/summary")
    page = r.json()
    res_title = page["title"]
    res_sum = page["extract"]
    return jsonify(context='\n'.join([res_title, res_sum]))




def predict_from_text_squad(input):
    squad_dict = convert_text_input_to_squad(input)
    return package_squad_prediction(squad_dict)

def predict_from_file_squad(input):
    try:
        squad_dict = convert_file_input_to_squad(input)
    except AssertionError:
        return []
    return package_squad_prediction(squad_dict)

def predict_from_input_squad(context, questions, id):
    squad_dict = convert_context_and_questions_to_squad(context, questions)
    return package_squad_prediction(squad_dict, id)

def package_squad_prediction(squad_dict, id="context-default"):
    prediction, dt = evaluate_input(squad_dict)
    packaged_predictions = []
    highlight_script = ""
    for entry in squad_dict["data"]:
        title = entry["title"]
        inner_package = []
        for p in entry["paragraphs"]:
            context = p["context"]
            qas = [(q["question"], prediction[q["id"]][0],
                    datetime.datetime.now().strftime("%d %B %Y %I:%M%p"),
                    "%0.02f seconds" % (dt),
                    '#' + id,
                    generate_highlight(context, id, prediction[q["id"]][1], prediction[q["id"]][2])) for q in p["qas"]]
            if not highlight_script:
                highlight_script = qas[0][5]
            inner_package.append((context, qas))
        packaged_predictions.append((title, inner_package))
    return packaged_predictions, highlight_script

def generate_highlight(context, id, start_index, stop_index):
    if start_index > -1:
        context_split = context.split()
        start_index = len(" ".join(context_split[:start_index]))
        stop_index = len(" ".join(context_split[:stop_index + 1]))
    return 'highlight(' + '"#' + id + '",' + str(start_index) + ',' + str(stop_index) + ');return false;'

def evaluate_input(squad_dict, passthrough=False):
    args.input_data = squad_dict
    t = time.time()
    predictions = evaluate(args, model, tokenizer)
    dt = time.time() - t
    app.logger.info("Loading time: %0.02f seconds" % (dt))
    if passthrough:
        return predictions, squad_dict, dt
    return predictions, dt

@app.route('/_input_helper')
def input_helper():
    # print(session['context'])
    context = session['context'][-1]
    text = context[0]
    id = context[1]
    # print(text)
    questions = unquote(request.args.get("question_data", "", type=str)).strip()
    app.logger.info("input text: {}\n\nquestions:{}".format(text, questions))
    predictions, highlight = predict_from_input_squad(text, questions, id)
    if text and questions:
        return jsonify(result=
                       render_template('live_results.html',
                                       predict=predictions),
                       highlight_script=highlight)
    return jsonify(result="")
    # else:
    #     if "file_urls" not in session or session['file_urls'] == []:
    #         return redirect(url_for('index'))
    #     file_urls = session['file_urls']
    #     session.pop('file_urls', None)
    #     app.logger.info("input file list: {}".format(file_urls))
    #     return jsonify(result=
    #                    render_template('results.html',
    #                                    file_urls=file_urls,
    #                                    predict=predict_from_file_squad))
@app.route('/_store_context')
@delay_func
def store_context():
    text = unquote(request.args.get("text_data", "", type=str)).strip()
    app.logger.info("input text: {}".format(text))

    remove_files = False
    if not text:
        if "file_urls" not in session or session['file_urls'] == []:
            return redirect(url_for('index'))
        file_urls = session['file_urls']
        session.pop('file_urls', None)
        with open(file_urls[-1], "r") as f:
            text = f.read()
        for f in file_urls:
            if os.path.exists(f):
                os.remove(f)
        remove_files = True

    if text:
        # if "context" not in session:
        session['context'] = []
        split_text = text.split("\n", 1)
        if len(split_text) > 1:
            # print(session['context'])
            curr_id = str(uuid.uuid4().hex[:8])
            session['context'].append((text, curr_id))
            session.modified = True
            return jsonify(title=split_text[0].strip(),
                           context=render_template("unique_context.html",
                                                    unique_id=curr_id,
                                                    content=re.sub("\n+", "\n", split_text[1].strip())),
                           clear_files=remove_files)
        else:
            return jsonify(context="")



if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=PORT)