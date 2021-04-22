"""A setuptools based setup module. 
    They say, "Always prefer setuptools over distutils

    See:
    https://github.com/pypa/sampleproject
    https://github.com/pypa/sampleproject/blob/master/setup.py
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding='utf-8')

setup(
    name='astrobf',
    version='0.0.1',
    description='Tone mapping for astronomical images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Hoseung/astroBF',
    classifiers=['Programming Language :: Python :: 3'],
    python_requires='>=3.7',
    install_requires=['numpy', 'scikit-image', 'astropy', 'sklearn', 'matplotlib'],
    extras_require={'dev':[''],
                    'test':['']},
)
