from dataclasses import dataclass, fields
from typing import Any, Union

CONTAINER_REGISTRY = dict()

def register_container(cls):
    CONTAINER_REGISTRY[cls.__name__] = cls
    return cls

def container_factory(container_name, values):
    if container_name in CONTAINER_REGISTRY:
        container_cls = CONTAINER_REGISTRY[container_name]
        new_values = convert_dictvalues(values, container_cls)
        return container_cls(**new_values)
    else:
        raise ValueError(f"Container '{container_name}' is not registered in the factory.")

def convert_dictvalues(d: dict, cls: Any) -> dict:
    converted = {}
    for field in fields(cls):
        field_name = field.name
        field_type = field.type
        if field_name in d:
            converted[field_name] = convert_value(d[field_name], field_type)
        else:
            raise ValueError(f"Field '{field_name}' is missing in the input dictionary.")
    return converted

def convert_value(value: str, type_hint: Any) -> Any:
    if type_hint == int:
        return int(value)
    elif type_hint == float:
        return float(value)
    elif type_hint == str:
        return value
    else:
        raise ValueError(f"Unsupported type: {type_hint}")

@register_container
@dataclass
class C1:
    """ """

    a: int
    b: str #Union[str, list[str]]
    
    # def __post_init__(self):
    #     """ """

    #     if isinstance(self.ENGINE, str):
    #         setattr(self, self.ENGINE, factory_engines(self.ENGINE, self.ENGINE_SETT))
    #     elif isinstance(self.ENGINE, list):
    #         for i, engine_i in  enumerate(self.ENGINE):
    #             setattr(self, engine_i, factory_engines(engine_i, self.ENGINE_SETT[i]))


@register_container
@dataclass
class Mc1:
    """ """

    spot: float
    vol: float #Union[str, list[str]]
    rate: float
    num_paths: int
        
    # def __post_init__(self):
    #     """ """

    #     if isinstance(self.ENGINE, str):
    #         setattr(self, self.ENGINE, factory_engines(self.ENGINE, self.ENGINE_SETT))
    #     elif isinstance(self.ENGINE, list):
    #         for i, engine_i in  enumerate(self.ENGINE):
    #             setattr(self, engine_i, factory_engines(engine_i, self.ENGINE_SETT[i]))
    
