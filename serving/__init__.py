# coding=utf-8
# @FileName     :__init__.py.py
# @DateTime     :2022/3/9 18:41
# @Author       :Haoguo Feng
import logging
from flask import Flask, request, jsonify
from serving.inference.entity_server import extract_entity, model, train_args, tag_to_id, id_to_tag, tf_config, sess
from jinja2 import Environment, Template, FileSystemLoader

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
logging.getLogger().setLevel(logging.DEBUG)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

env = Environment(loader=FileSystemLoader('serving/template'))
template = env.get_template('base.html')


@app.route("/")
def default():
    return template.render()


@app.route("/entity", methods=['GET', 'POST'])
def entity():
    if request.method == 'GET':
        return template.render()
    else:  # request.method == 'POST':
        if request.content_type.startswith('application/json'):
            content = request.json.get('content')
        elif request.content_type.startswith('multipart/form-data'):
            content = request.form.get('content')
        else:
            content = request.values.get('content')
        per, loc, org = extract_entity(content)

        return template.render(per=per, loc=loc, org=org, content=content)


if __name__ == "__main__":
    run_code = 0
