from typing import Literal

type Reduction = Literal["mean", "sum", "max", "min", "prod"]
type Backend = Literal["auto", "torch", "triton", "torch_eager"]
