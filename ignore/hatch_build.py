from typing import * 
from extension.runner import BuildRunner
from extension.interface import ExtensionModules

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomHook(BuildHookInterface):
    def initialize(self, version, build_data):
        print("help")
        if self.target_name != 'wheel' or version != 'editable':
            return

# ExtensionModules(name: str, root: str, metadata: dict, config: dict)
class ExampleExtensionModules(ExtensionModules):
    def __init__(self, name: str, root: str, metadata: dict, config: dict):
      print("testing")
      super().__init__(name, root, metadata, config)
      runner = BuildRunner('.', config['project'])
      runner.generate_inputs(config)
      runner.generate_outputs(config)

    def inputs():
      return(["src/geom_dtn/ext/"])

    def outputs():
      return([])

