"""A module for defining types and type aliases used in the project."""
from typing import Annotated, Literal, Sequence, TypeAlias, TypedDict
import numpy as np
import numpy.typing as npt

Color: TypeAlias = tuple[float, float, float, float]
ColorInt: TypeAlias = tuple[int, int, int, int]
ColorLike: TypeAlias = Color | Annotated[npt.NDArray[np.double], Literal[4]]
ColorsLike: TypeAlias = Color | Annotated[npt.NDArray[np.double], Literal['N', 4]] | list[Color]
Limb: TypeAlias = tuple[Annotated[Sequence[int], 2], Color]

Skeleton: TypeAlias = dict[int, Annotated[npt.NDArray[np.double], Literal[2]]]
SkeletonValidity: TypeAlias = dict[int, bool]
SkeletonWeights: TypeAlias = dict[int, Annotated[npt.NDArray[np.double], Literal['N']]]
SkeletonPath: TypeAlias = dict[int, Annotated[npt.NDArray[np.double], Literal[81]]]

class FitResult(TypedDict):
    """A type for the result of a fit."""
    theta: npt.NDArray[np.double]
    pi: npt.NDArray[np.double]
    designMatrix: npt.NDArray[np.double]
    diff: float
    convergence: bool
    responsibilities: npt.NDArray[np.double]
    niters: list[int]

FitResults: TypeAlias = dict[int, FitResult]
