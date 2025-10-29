from enum import Enum


class Models(Enum):
    K_IBL = 'k_ibl'
    FW_K_IBL = 'fw_k_ibl'
    IR_K_IBL = 'ir_k_ibl'
    SVM = 'svm'

    def __str__(self):
        return self.value
