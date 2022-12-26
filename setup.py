from setuptools import setup, find_packages
import os

VERSION = '0.0.1'
DESCRIPTION = 'Short - term Prediction of Poor Data Using Grey Prediction Model'

setup(
    name="greypredict",
    version=VERSION,
    author="Fu Long",
    author_email="15045435783.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=open('README.md', encoding="UTF8").read(),
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'grey theory', 'grey predict'],
    data_files=[],
    entry_points={
    'console_scripts': [

    ]
    },
    license="HIT",
    url="https://github.com/FU-Long12138/Grey-Predict",
    scripts=['scripts/demo.py'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows"
    ]
)