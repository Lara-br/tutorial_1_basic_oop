import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


import random
import math
import matplotlib.pyplot as plt
from statistics import mean

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    @classmethod
    def random(self, x_min: float = 0, x_max: float = 1, y_min: float = 0, y_max: float = 1):
        return Point(random.uniform(x_min, x_max),
                     random.uniform(y_min, y_max))
    
    @classmethod
    def mean(self, points):
        x = mean(p.x for p in points)
        y = mean(p.y for p in points)
        return Point(x, y)
    
def ccw_order(points):
    cx = sum(p.x for p in points) / len(points)
    cy = sum(p.y for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p.y - cy, p.x - cx))

class Fractal(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    def plot(self) -> None:
        try:
            plt.scatter([p.x for p in self._points], [p.y for p in self._points],  s=1, color='black')
            plt.show()
        except AttributeError:
            raise AttributeError("Fractal needs a self.points list[Point] to be plotted.")
    
    @property
    def points(self):
        try:
            return self._points.copy()
        except AttributeError:
            return None
    
class SierpinskyPolygon(Fractal):
    def __init__(self, corners: list[Point], depth: int):
        self._corners = ccw_order(corners.copy())
        self._points = corners.copy()

        self._last_new = [Point.mean([self.corners[0], self.corners[1]])]
        for _ in range(depth):
            self.generate_polygon()
        
            
    def generate_polygon(self) -> None:  
        new_points = []      
        for p1 in self._last_new:
            for p2 in self.corners:
                new_points.append(Point.mean([p1, p2]))
        self._points += new_points
        self._last_new = new_points.copy()

class MandelBrot(Fractal):
    def __init__(self, size: int, depth: int):
        self._depth = depth
        self._points = []

        while len(self._points) < size:
            c = random.uniform(-2, 0.5) + 1j*random.uniform(-1.1, 1.1) # Magic numbers are bounds, efficiency depends on how close to true shape of set.
            if self.check_sequence(c):
                self._points.append(Point(c.real, c.imag))

    def check_sequence(self, c) -> bool:
        z = 0 + 0j
        for i in range(self._depth):
            z = z * z + c
            if abs(z) > 2:
                return False
        return True
    
if __name__ == "__main__":
    points = [Point.random() for _ in range(3)]
    
    sier_tri = SierpinskyPolygon(points, 4)
    sier_tri.plot()
    
    mand = MandelBrot(10000, 20)
    mand.plot()
    
    