import torch as th
from flask import Flask
from flask import request
from trainer import ema
import glob, os
import json

app = Flask(__name__)

@app.route('/plot/', methods=['GET', 'POST'])
def plot():
    paths  = json.loads(request.args.get('paths'))
    graphs = json.loads(request.args.get('graphs'))
    num_of_points = json.loads(request.args.get('num_of_points'))
    if num_of_points == 0: num_of_points = 100
    
    data = {}
    for path in paths:
        data[path] = th.load("saved/" + path + "/trainer")
        
    def getValue(d, graph, alpha=0.05):
        value = getattr(d, graph)
        return ema(value if graph == "losses" else th.cat(value)[:, -1], alpha)
        
    lengths = {}
    for graph in graphs:
        max_length = 0
        for path, d in data.items():
            length = len(getValue(d, graph))
            if max_length < length: max_length = length
        lengths[graph] = max_length
        
    output = {}
    for graph in graphs:
        graph_data = {}
        step = round(lengths[graph]/num_of_points)
        for path, d in data.items():
            graph_data[path] = getValue(d, graph)[::step]
        output[graph] = graph_data
    return json.dumps(output)

@app.route('/trainers/')
def trainings():
    files = []
    for filename in glob.iglob('saved/**/trainer', recursive=True):
        fn = os.path.dirname(filename)
        files.append("/".join(fn.strip("/").split('/')[1:]))
    return json.dumps(files)

@app.route('/')
def main():
    return open('board.html', 'r').read()

if __name__ == '__main__':
    app.run()