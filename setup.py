#!/usr/bin/env python
# flake8: noqa
# -*- coding: utf-8 -*-

# Note: To use the "upload" functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
import subprocess
from shutil import rmtree

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "garrus"
DESCRIPTION = "Garrus. Python framework for better confidence estimate of deep neural networks."
URL = "https://github.com/sleep3r/garrus"
EMAIL = "sleep3r@icloud.com"
AUTHOR = "Alexander Kalashnikov"
REQUIRES_PYTHON = ">=3.7.0"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename):
    """Docs? Contribution is welcome."""
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()


def load_readme():
    """Docs? Contribution is welcome."""
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    with io.open(readme_path, encoding="utf-8") as f:
        return f"\n{f.read()}"


def load_version():
    """Docs? Contribution is welcome."""
    context = {}
    with open(os.path.join(PROJECT_ROOT, "garrus", "__init__.py")) as f:
        exec(f.read(), context)
    return context["__version__"]


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def run(self):
        """Docs? Contribution is welcome."""
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(PROJECT_ROOT, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        subprocess.call("{0} setup.py sdist bdist_wheel --universal".format(sys.executable), shell=False)

        self.status("Uploading the package to PyPI via Twine…")
        subprocess.call("twine upload dist/*", shell=False)

        self.status("Pushing git tags…")
        subprocess.call("git tag v{0}".format(load_version()), shell=False)
        subprocess.call("git push --tags", shell=False)

        sys.exit()


setup(
    name=NAME,
    version=load_version(),
    description=DESCRIPTION,
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    keywords=[
        "Machine Learning",
        "Distributed Computing",
        "Deep Learning",
        "Reinforcement Learning",
        "Computer Vision",
        "Natural Language Processing",
        "Recommendation Systems",
        "Information Retrieval",
        "PyTorch",
        "Confidence calibration",
        "Confidence ranking"
    ],
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    download_url=URL,
    project_urls={
        "Bug Tracker": URL + "/issues",
        "Documentation": URL + "/wiki",
        "Source Code": URL ,
    },
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        # Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Programming
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    # $ setup.py publish support.
    cmdclass={"upload": UploadCommand},
)
