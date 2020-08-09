import abc


class Prior(abc.ABC):
    @abc.abstractclassmethod
    def transform_from_unit_cube(self, cube):
        pass


class LinearPrior(Prior):
    def __init__(self, bounds):
        self.bounds = bounds

    def transform_from_unit_cube(self, cube):
        denormalized = []
        for i, b in enumerate(self.bounds):
            denormalized.append(cube[i] * (b[1] - b[0]) + b[0])
        return denormalized
