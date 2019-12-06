from distutils.core import setup
setup(
  name = "karateclub",
  packages = ["karateclub"],
  version = "0.6",
  license = "MIT",
  description = "A general purpose library for community detection and graph clustering research.",
  author = "Benedek Rozemberczki",
  author_email = "benedek.rozemberczki@gmail.com",
  url = "https://github.com/benedekrozemberczki/karateclub",
  download_url = "https://github.com/benedekrozemberczki/karateclub/archive/v_06.tar.gz",
  keywords = ["community", "detection", "networkx", "graph", "clustering"],
  install_requires=[
          "numpy",
          "networkx",
          "tqdm",
          "python-louvain",
      ],
  classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Build Tools",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.5",
  ],
)
