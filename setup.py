"""
    Setup file for soda-data.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.4.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    try:
        setup(
            use_scm_version={"version_scheme": "no-guess-dev"},
            name="soda_model",
            version="1.0.0",
            python_requires=">=3.8",
            author="Source Data",
            author_email="source_data@embo.org",
            description="""Biomedical natural language processing models to generate digital knowledge from scientific publications.""",
            long_description=long_description,
            long_description_content_type="text/markdown",
            url="https://github.com/source-data/soda-model",
            packages=["soda_model"],
            install_requires=[
                # "torch~=1.12.0",
                "torch==1.11.0a0+bfe5ad2",
                "transformers~=4.20",
                "datasets~=2.10.0",
                "scikit-learn",
                "python-dotenv",
                "seqeval",
                "wandb<0.13.0",
                "allennlp==2.10.1"
            ],
            dependency_links=["https://download.pytorch.org/whl/cu101"],
            classifiers=[
                # full list: https://pypi.org/pypi?%3Aaction=list_classifiers
                "Development Status :: 1 - Planning",
                "Intended Audience :: Science/Research",
                "Programming Language :: Python :: 3.6",
                "License :: Other/Proprietary License",
                "Operating System :: MacOS :: MacOS X",
                "Operating System :: POSIX",
                "Topic :: Scientific/Engineering :: Artificial Intelligence",
                "Topic :: Scientific/Engineering :: Bio-Informatics",
                "Topic :: Software Development :: Libraries",
            ],
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
