from enum import Enum


class GlueTaskEnum(Enum):
    MNLI = "MNLI"
    QQP = "QQP"
    QNLI = "QNLI"
    SST_2 = "SST-2"
    CoLA = "CoLA"
    STS_B = "STS-B"
    MRPC = "MRPC"
    RTE = "RTE"
    WNLI = "WNLI"
