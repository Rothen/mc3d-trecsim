"""A module for visualizing measurements."""
from __future__ import annotations
from typing import Annotated, Sequence

import numpy as np

from measurement import Measurement, ScalarMeasurement

class MeasurementVisualizer:
    """A class for visualizing measurements."""
    def __init__(self,
                 measurement: Measurement,
                 label: str = '',
                 unit: str = '',
                 color: Annotated[Sequence[int], 4] | None = None,
                 font_size: int = 12
                 ):
        """Initializes the MeasurementVisualizer class.

        Args:
            measurement (Measurement): The measurement to visualize.
            label (str, optional): The label of the measurement. Defaults to ''.
            unit (str, optional): The unit of the measurement. Defaults to ''.
            color (Annotated[Sequence[int], 4] | None, optional): The color of the measurement.
                Defaults to None.
            font_size (int, optional): The font size of the measurement. Defaults to 12.
        """
        self.measurement = measurement
        self.label: str = '%s ' % (label, ) if len(label) > 0 else ''
        self.unit: str = ' %s' % (unit, ) if len(unit) > 0 else ''
        self.color: Annotated[Sequence[int],
                              4] = color if color is not None else [255]*4
        self.font_size: int = font_size

    def visualize(self) -> list[str]:
        return [
                '%s%.1f%s' % (self.label, value, self.unit)
                if isinstance(self.measurement, ScalarMeasurement)
                else '%s%.1f, %.1f, %.1f%s' % (self.label, value[0], value[1], value[2], self.unit)
            for value in self.measurement.values]
