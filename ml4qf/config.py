import ml4qf.containers as containers
from . import parser #import ml4qf.parser as parser

class Config:

    def __init__(self, settings):
        self.__build(settings)
        
    def __build(self, settings: dict):
        for ci, di in settings.items():
            if isinstance(di, dict):
                setattr(self, ci, containers.container_factory(ci, di))
            else:
                setattr(self, ci, di)

    @classmethod
    def _buildConfig_txt(cls, inputfile):
        settings = parser.parse_textfile(inputfile)
        return cls(settings)


config = Config._buildConfig_txt('/home/acea/projects/XQFM/examples/prototyping/joshicpp/input4.txt')    
