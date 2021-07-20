from setuptools import setup, find_packages

setup(
    name='interimit',
    version='0.0.1',
    url='https://github.com/sisl/InteractionImitation/',
    author='Arec Jamgochian',
    author_email='jamgochian95@gmail.com',
    description='Imitation Learning on INTERACTION Dataset',
    packages=find_packages(),
    install_requires=[
        'tikzplotlib',
        'torch',
        'pytest',
        'json5',
        'gym',
        'intersim @ git+https://github.com/sisl/InteractionSimulator',
    ],
)
