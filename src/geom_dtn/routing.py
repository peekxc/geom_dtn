import os

def load_satellites():
  # 'starlink.txt'
  import importlib.resources as pkg_resources
  from geom_dtn import data as package_data_mod
  # starlink = pkg_resources.read_text(package_data_mod, 'starlink.txt')
  from skyfield.api import load, wgs84
  satellites = load.tle_file(os.path.join(package_data_mod.__path__._path[0], 'starlink.txt'))
  return(satellites)