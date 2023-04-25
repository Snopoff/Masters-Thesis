from setuptools import setup, find_packages

setup(
    name="Topology and NNs experiments",
    version="0.0.1",
    author="Paul Snopov",
    author_email="snopov.pm@phystech.com",
    description="A package containing the experiments for my masters thesis",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
