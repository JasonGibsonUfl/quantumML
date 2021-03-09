from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="quantumML",
    version="0.0.36",
    author="Jason Gibson",
    author_email="jasongibson@ufl.edu",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jasongibsonufl/quantumML",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    package_data={
        "":["*.txt"],
    },
    include_package_data=True,
)
