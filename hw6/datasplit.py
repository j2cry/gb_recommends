import numpy as np
import pandas as pd


class DataSplit:
    def __init__(self, data, field, dimensions):
        """ Split data to parts according to the specified dimensions by the value of the field (from more to less)
        :param data: source dataset
        :param field: target field to split
        :param dimensions: final dimensions of data parts
        """
        assert data is not None and isinstance(data, pd.DataFrame), '`data` must be a DataFrame'
        assert field in data.columns, '`data` must contain the given field column'
        assert len(dimensions) > 0, '`sizes` cannot be empty'
        self.data = data
        self.field = field
        self.dimensions = dimensions
        self.__split()

    def __split(self):
        """ Split data """
        max_value = self.data[self.field].max()
        mask0 = self.data[self.field] <= max_value - np.sum(self.dimensions)
        setattr(self, 'part0', mask0)
        for i in range(len(self.dimensions)):
            lower = max_value - np.sum(self.dimensions[i:])
            upper = lower + self.dimensions[i]
            mask = (self.data[self.field] > lower) & (self.data[self.field] <= upper)
            setattr(self, f'part{i + 1}', mask)