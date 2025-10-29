from enum import Enum


class NormalizationStrategy(Enum):
    MEAN_NORMALIZE = 'mean_normalize'
    STANDARDIZE = 'standardize'
    UNIT_VECTOR = 'unit_vector'
    MINMAX_SCALING = 'minmax_scaling'

    def __str__(self):
        return self.value


class EncodingStrategy(Enum):
    LABEL_ENCODE = 'label_encode'
    ONE_HOT_ENCODE = 'one_hot_encode'

    def __str__(self):
        return self.value


class MissingValuesNumericStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    ZERO = 'zero'
    DROP = 'drop'
    MODEL = 'model'

    def __str__(self):
        return self.value


class MissingValuesCategoricalStrategy(Enum):
    MODE = 'mode'
    CONSTANT = 'constant'
    DROP = 'drop'

    def __str__(self):
        return self.value


class RetentionPolicy(Enum):
    NEVER_RETAIN = 'never_retain'
    ALWAYS_RETAIN = 'always_retain'
    DIFFERENT_CLASS_RETENTION = 'different_class_retention'
    DD_RETENTION = 'DD_retention'

    def __str__(self):
        return self.value


class FeatureWeightingMethod(Enum):
    RELIEFF = 'relieff'
    INFORMATION_GAIN = 'information_gain'

    def __str__(self):
        return self.value
