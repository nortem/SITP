import pathlib

import yaml
from pydantic import BaseModel, validator

from utils.config_validation import Environment


class Algorithm(BaseModel):
    name: str = None
    memory_limit: int = None
    path_to_weights: str = None
    full_name: str = None
    device: str = 'cpu'

    @validator('full_name')
    def create_full_name(cls, v, values):
        if v:
            return v
        if values['name'] and values['path_to_weights']:
            return values['name'] + '-' + pathlib.Path(values['path_to_weights']).name
        return values['name']

    def __repr__(self):
        return ":".join(map(str, [key for key in [self.name, self.path_to_weights, self.memory_limit] if key]))


class TabulateConfig(BaseModel):
    drop_keys: list = ['seed', 'flowtime']
    metrics: list = ['ISR', 'CSR', 'makespan', 'FPS', 'map_name']
    round_digits: int = 2


class EvaluationConfig(BaseModel, ):
    name: str = None
    environment: Environment = Environment()
    algo: Algorithm = None
    resolved_vars: dict = None
    results: dict = None
    id: int = None
    tabulate_config: TabulateConfig = TabulateConfig()


def main():
    file_name = "evaluation/map2config/street-map-configs/Berlin_0_256.map.yaml"
    with open(file_name) as f:
        config = yaml.safe_load(f)
        q = EvaluationConfig(**config)
        print(q)


if __name__ == '__main__':
    main()
