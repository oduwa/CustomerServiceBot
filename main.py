from flask import Flask
from flask import request,render_template
import json
from IntentDetector import IntentDetector
from InfoExtractor import InfoExtractor

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('task1_gui.html')

@app.route("/task2")
def task2():
    return render_template('task2_gui.html')

@app.route("/intent", methods=['GET', 'POST'])
def intent():
    print request.args
    query_message = request.args.get('message')
    if query_message:
        result = IntentDetector().classify_message_intent({'text':query_message, 'source':'customer'})
        return json.dumps(result)
    else:
        return 'ERROR: Request must contain "message" parameter.'

@app.route("/extract", methods=['GET', 'POST'])
def extract():
    query_message = request.args.get('message')
    if query_message:
        result = InfoExtractor().extract_info(query_message)
        return json.dumps(result)
    else:
        return 'ERROR: Request must contain "message" parameter.'

if __name__ == "__main__":
    app.run(debug=True)
