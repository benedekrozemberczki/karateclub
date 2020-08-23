from setuptools import find_packages, setup

install_requires = ["numpy",
                    "networkx",
                    "tqdm",
                    "python-louvain",
                    "scikit-learn",
                    "scipy",
                    "pygsp",
                    "gensim",
                    "pandas",
                    "six"]


setup_requires = ['pytest-runner']


tests_require = ['pytest',
                 'pytest-cov',
                 'mock']


keywords = ["community",
            "detection",
            "networkx",
            "graph",
            "clustering",
            "embedding",
            "deepwalk",
            "graph2vec",
            "node2vec",
            "deep",
            "learning",
            "louvain",
            "machine-learning",
            "deep-learning",
            "deeplearning"]


setup(
  name = "karateclub",
  packages = find_packages(),
  version = "1.0.14",
  license = "GPLv3",
  description = "A general purpose library for community detection, network embedding, and graph mining research.",
  author = "Benedek Rozemberczki",
  author_email = "benedek.rozemberczki@gmail.com",
  url = "https://github.com/benedekrozemberczki/karateclub",
  download_url = "https://github.com/benedekrozemberczki/karateclub/archive/v_10014.tar.gz",
  keywords = keywords,
  install_requires = install_requires,
  setup_requires = setup_requires,
  tests_require = tests_require,
  classifiers = ["Development Status :: 3 - Alpha",
                 "Intended Audience :: Developers",
                 "Topic :: Software Development :: Build Tools",
                 "License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3.7"],
)
