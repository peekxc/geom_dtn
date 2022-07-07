# -*- coding: utf-8 -*-
import os 
import sys 
import pathlib
import importlib
import glob
import shutil
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
import platform

suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

package_dir = \
{'': 'src'}

packages = \
['geom_dtn', 'geom_dtn.extensions']

package_data = \
{'': ['*'], 'geom_dtn.extensions': ['*.so', '*.pyd', 'extensions/*.so', 'extensions/*.pyd'] }

install_requires = \
['numpy>=1.21.3,<2.0.0', 'scipy>=1.6']

# From: https://stackoverflow.com/questions/51108256/how-to-take-a-pathname-string-with-wildcards-and-resolve-the-glob-with-pathlib
def expandpath(path_pattern):
	p = Path(path_pattern).expanduser()
	parts = p.parts[p.is_absolute():]
	return Path(p.root).glob(str(Path(*parts)))

class CustomExtension(Extension):
  def __init__(self, path):
    self.path = path
    super().__init__(pathlib.PurePath(path).name, [])

class build_CustomExtensions(build_ext):
  def run(self):
    for ext in (x for x in self.extensions if isinstance(x, CustomExtension)):
      source = f"{ext.path}{suffix}"
      build_dir = pathlib.PurePath(self.get_ext_fullpath(ext.name)).parent
      os.makedirs(f"{build_dir}/{pathlib.PurePath(ext.path).parent}", exist_ok = True)
      shutil.copy(f"{source}", f"{build_dir}/{source}")

def find_extensions(directory):
  extensions = []
  for path, _, filenames in os.walk(directory):
    for filename in filenames:
      filename = pathlib.PurePath(filename)
      if pathlib.PurePath(filename).suffix == suffix:
        extensions.append(CustomExtension(os.path.join(path, filename.stem)))
  return extensions

setup_kwargs = {
    'name': 'geom_dtn',
    'version': '0.2.2',
    'description': 'Geometry-aware Delay Tolerant Networking',
    'author': 'Matt Piekenbrock',
    'author_email': 'matt.piekenbrock@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/peekxc/geom_dtn',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.8,<3.10',
    'ext_modules': find_extensions("src/geom_dtn"),
    'cmdclass': {'build_ext': build_CustomExtensions},
    # 'distclass': CustomDistribution
}

# Build first, then invoke setup 
build_extensions(setup_kwargs)
setup(**setup_kwargs)

