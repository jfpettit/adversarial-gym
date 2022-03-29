from os import PRIO_PGRP
import numpy as np
from abc import ABC


class Distribution(ABC):
    def __init__(self, seed: int) -> None:
        self.total_samples = 0
        self.sample_fn_count = 0
        np.random.seed(seed=seed)

    def sample(self) -> np.ndarray:
        pass

    @property
    def dist_info(self):
        pass

    def __repr__(self):
        string = "=== DIST INFO ===\n"
        for k, v in self.dist_info.items():
            string += f"{k}: {v}\n"
        return string

    def reset(self):
        self.total_samples = 0
        self.sample_fn_count = 0
    
    @property
    def expected_value(self) -> float:
        pass


class NormalDistribution(Distribution):
    def __init__(self, loc: float, scale: float, seed: int = 0) -> None:
        super().__init__(seed)
        self.loc = loc
        self.scale = scale

    def sample(self, size: tuple) -> np.ndarray:
        self.sample_fn_count += 1
        self.total_samples += np.prod(size)
        return np.random.normal(loc=self.loc, scale=self.scale, size=size)

    @property
    def dist_info(self):
        info = {
            "distribution type": self.__class__.__name__,
            "mean (aka expected value)": self.expected_value,
            "std": self.scale,
            "var": self.variance,
            "total samples drawn": self.total_samples,
            "times sample fcn has been called": self.sample_fn_count
        }
        return info

    @property
    def expected_value(self) -> float:
        return self.loc

    @property
    def variance(self) -> float:
        return np.sqrt(self.scale)


class UniformDistribution(Distribution):
    def __init__(self, low: int, high: int, seed: int = 0) -> None:
        super().__init__(seed)
        self.low = low
        self.high = high

    def sample(self, size: tuple) -> np.ndarray:
        self.sample_fn_count += 1
        self.total_samples += np.prod(size)
        return np.random.uniform(low=self.low, high=self.high, size=size)

    @property
    def dist_info(self):
        info = {
            "distribution type": self.__class__.__name__,
            "low": self.low,
            "high": self.high,
            "expected value": self.expected_value,
            "total samples drawn": self.total_samples,
            "times sample fcn has been called": self.sample_fn_count
        }
        return info

    @property
    def expected_value(self) -> float:
        return (self.high - self.low)/2


class DiscreteUniformDistribution(UniformDistribution):
    def __init__(self, low: int, high: int, seed: int = 0) -> None:
        super().__init__(low, high, seed)
    
    def sample(self, size: tuple) -> np.ndarray:
        self.sample_fn_count += 1
        self.total_samples += np.prod(size)
        return np.random.randint(low=self.low, high=self.high, size=size)


class BetaDistribution(Distribution):
    def __init__(self, a: float, b: float, seed: int = 0) -> None:
        super().__init__(seed)
        self.a = a
        self.b = b

    def sample(self, size: tuple) -> np.ndarray:
        self.sample_fn_count += 1
        self.total_samples += np.prod(size)
        return np.random.beta(a=self.a, b=self.b, size=size)

    @property
    def dist_info(self):
        info = {
            "distribution type": self.__class__.__name__,
            "a": self.a,
            "b": self.b,
            "expected value": self.expected_value,
            "variance": self.variance,
            "total samples drawn": self.total_samples,
            "times sample fcn has been called": self.sample_fn_count
        }
        return info
    
    @property
    def expected_value(self) -> float:
        return (1/(1 + (self.b/self.a)))

    @property
    def variance(self) -> float:
        return (self.a*self.b)/((self.a + self.b)**2 * (self.a + self.b + 1))


class PoissonDistribution(Distribution):
    def __init__(self, lam: float, seed: int) -> None:
        super().__init__(seed)
        self.lam = lam

    def sample(self, size: tuple) -> np.ndarray:
        self.sample_fn_count += 1
        self.total_samples += np.prod(size)
        return np.random.poisson(lam=self.lam, size=size)

    @property
    def expected_value(self) -> float:
        return self.lam

    @property
    def variance(self) -> float:
        return self.lam