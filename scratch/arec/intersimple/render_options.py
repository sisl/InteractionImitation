import sys
sys.path.append('../../../')
from src.util import render_env
ALL_OPTIONS = [(v,t) for v in [0,2,4,6,8] for t in [5, 10]]

def render_wrapper(**kwargs):
    render_env(**kwargs, options_list=ALL_OPTIONS)

if __name__=='__main__':
    import fire
    fire.Fire(render_wrapper)