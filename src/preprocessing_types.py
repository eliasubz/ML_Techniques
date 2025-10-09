from enum import Enum

class NormalizationStrategy(Enum):
    MEAN_NORMALIZE = 'mean_normalize'
    STANDARDIZE = 'standardize'
    UNIT_VECTOR = 'unit_vector'
    MINMAX_SCALING = 'minmax_scaling'

class EncodingStrategy(Enum):
    LABEL_ENCODE = 'label_encode'
    ONE_HOT_ENCODE = 'one_hot_encode'

class MissingValuesNumericStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    ZERO = 'zero'
    DROP = 'drop'
    MODEL = 'model'

class MissingValuesCategoricalStrategy(Enum):
    MODE = 'mode'
    CONSTANT = 'constant'
    DROP = 'drop'
