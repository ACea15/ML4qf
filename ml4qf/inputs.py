from dataclasses import dataclass, replace
from typing import Union

list_engines = ['keras', 'scikit']

def factory_engines(engine_type, settings):
    """


    Parameters
    ----------
    engine_type : 
    settings : 

    Returns
    -------
    out : 

    """
    

    assert engine_type in list_engines, "input engine not implemented"
    engine = globals()['Input_' + engine_type]
    return engine(settings)

@dataclass
class Input:
    """ """

    FIT_SETT: dict
    ENGINE: Union[str, list[str]]
    ENGINE_SETT: Union[dict, list[dict]]
    
    def __post_init__(self):
        """ """

        if isinstance(self.ENGINE, str):
            setattr(self, self.ENGINE, factory_engines(self.ENGINE, self.ENGINE_SETT))
        elif isinstance(self.ENGINE, list):
            for i, engine_i in  enumerate(self.ENGINE):
                setattr(self, engine_i, factory_engines(engine_i, self.ENGINE_SETT[i]))

@dataclass
class Input_ticker:
    
    TICKER: str
    YEAR0: int
    MONTH0: int
    DAY0: int
    YEARS_MODELLING: float
    FEATURES: dict
    DAYS_YEAR: int = 365
    PRICE: str = 'Close'
    LOG_RETURN: bool = False
    def __post_init__(self):
        """ """
        
        self.NUM_DAYS = self.DAYS_YEAR * self.YEARS_MODELLING

    def clone(self, **kwargs):

        return replace(self, **kwargs)


@dataclass
class Input_keras:

    LAYERS: Union[tuple, str]
    KERAS_MODEL: str ='Sequential'
    OPTIMIZER_NAME: str='adam'
    LOSS_NAME: str='mse'
    METRICS: str=None
    OPTIMIZER_SETT: str =None
    COMPILE_SETT: str =None
    LOSS_SETT: str =None
    
    def __post_init__(self):

        if isinstance(self.LAYERS, dict):
            self.LAYERS = dict2tuple(self.LAYERS)
    
@dataclass
class Input_scikit:
    pass
