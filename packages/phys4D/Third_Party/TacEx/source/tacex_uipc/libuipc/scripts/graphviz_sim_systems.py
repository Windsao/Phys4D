import json
import os
import pathlib as pl
from graphviz import Digraph
import argparse

file_name = 'systems.json'


parser = argparse.ArgumentParser()
parser.add_argument('--file', help='file name')
args = parser.parse_args()

if args.file:
    file_name = args.file

if not os.path.exists(file_name):
    print(f'File {file_name} does not exist!')
    exit(1)


f = open(file_name, 'r')
systems = json.load(f)
f.close()

dot = Digraph(comment='The Round Table')


dot.attr(fontsize='40')

dot.attr(rankdir='LR')

for system in systems['sim_systems']:
    name = system['name'].replace('class uipc::backend::cuda::', '')
    deps = [dep.replace('class uipc::backend::cuda::', '') for dep in system['deps']]
    dot.node(name)
    for dep in deps:
        dot.edge(name, dep)

print(dot)