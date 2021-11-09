import sys
sys.path.append('../../../')
from src.util import render_env


if __name__=='__main__':
    import fire
    fire.Fire(render_env)