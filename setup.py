import os
from setuptools import setup, find_packages
import pathlib

root = pathlib.Path(__file__).parent.resolve()

def read(fname):
  return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = "geom_dtn",
  version = "0.0.1",
  author = "Matt Piekenbrock",
  author_email = "matt.piekenbrock@gmail.com",
  description = ("."),
  license = "Apache 2.0",
  package_dir={"": "src"},
  package_data={'geom_dtn': ['data/*.bsp', 'data/*.txt', 'data/*.csv']},
  packages=find_packages(where="src"),
  install_requires=["numpy"],
  python_requires=">=3.7, <4",
  long_description = (root / "README.md").read_text(encoding="utf-8")
)