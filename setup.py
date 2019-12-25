from distutils.core import setup
from setuptools import find_packages
setup(
  name = "karateclub",
  packages = find_packages(),
  version = "0.24",
  license = "MIT",
  description = "A general purpose library for community detection and graph clustering research.",
  author = "Benedek Rozemberczki",
  author_email = "benedek.rozemberczki@gmail.com",
  url = "https://github.com/benedekrozemberczki/karateclub",
  download_url = "https://github.com/benedekrozemberczki/karateclub/archive/v_024.tar.gz",
  keywords = ["community", "detection", "networkx", "graph", "clustering"],
  install_requires=[
          "numpy",
          "networkx",
          "tqdm",
          "python-louvain",
          "sklearn",
          "scipy",
          "pygsp",
      ],
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Build Tools",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.5",
  ],
)
