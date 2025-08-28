import math
import random
from abc import ABC, abstractmethod
from statistics import mean

import matplotlib.pyplot as plt


class Point:
    def __init__(self, x: float, y: float) -> None:
        """
        Initialize an object with x and y coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
        """
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        """
        Return a string representation of the Point object in the format 'Point(x, y)'.
        Returns:
            str: A string representing the Point instance with its x and y coordinates.
        """

        return f"Point({self.x}, {self.y})"

    @classmethod
    def random(cls, x_min: float = 0, x_max: float = 1, y_min: float = 0, y_max: float = 1,
    ) -> "Point":
        """
        Generates a random Point within the specified rectangular bounds.
        Args:
            x_min (float, optional): Minimum x-coordinate. Defaults to 0.
            x_max (float, optional): Maximum x-coordinate. Defaults to 1.
            y_min (float, optional): Minimum y-coordinate. Defaults to 0.
            y_max (float, optional): Maximum y-coordinate. Defaults to 1.
        Returns:
            Point: A Point object with random x and y coordinates within the given bounds.
        """
        return Point(random.uniform(x_min, x_max), random.uniform(y_min, y_max))

    @classmethod
    def mean(cls, points) -> "Point":
        """
        Calculates the mean (average) position of a collection of Point objects.
        Args:
            points (Iterable[Point]): An iterable of Point objects whose mean position is to be calculated.
        Returns:
            Point: A new Point instance representing the mean x and y coordinates of the input points.
        """

        x = mean(p.x for p in points)
        y = mean(p.y for p in points)
        return Point(x, y)


def ccw_order(points: list[Point]) -> list[Point]:
    """
    Sorts a list of Point objects in counter-clockwise (CCW) order around their centroid.

    Args:
        points (list[Point]): A list of Point objects to be sorted.

    Returns:
        list[Point]: The input points sorted in counter-clockwise order around their centroid.
    """
    cx = sum(p.x for p in points) / len(points)
    cy = sum(p.y for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p.y - cy, p.x - cx))


class Fractal(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    def plot(self) -> None:
        """
        Plots the fractal points using matplotlib.
        This method creates a scatter plot of the points stored in self._points,
        where each point is expected to have 'x' and 'y' attributes. If the points
        are not properly defined, raises an AttributeError with an informative message.
        Raises:
            AttributeError: If self._points is not a list of objects with 'x' and 'y' attributes.
        """

        try:
            plt.scatter(
                [p.x for p in self._points],
                [p.y for p in self._points],
                s=1,
                color="black",
            )
            plt.show()
        except AttributeError:
            msg = "Fractal needs a self.points list[Point] to be plotted."
            raise AttributeError(
                msg,
            )

    @property
    def points(self) -> list[Point] | None:
        """
        Returns a copy of the list of Point objects representing the fractal's points.
        Returns:
            list[Point] | None: A copy of the list of Point objects if available, otherwise None.
        """
        try:
            return self._points.copy()
        except AttributeError:
            return None


class SierpinskyPolygon(Fractal):
    def __init__(self, corners: list[Point], depth: int) -> None:
        """
        Initializes the fractal object with given polygon corners and recursion depth.
        Args:
            corners (list[Point]): The list of points representing the corners of the initial polygon.
            depth (int): The number of recursive steps to generate the fractal.
        """

        self._corners = ccw_order(corners.copy())
        self._points = corners.copy()

        self._last_new = [Point.mean([self.corners[0], self.corners[1]])]
        for _ in range(depth):
            self.generate_polygon()

    def generate_polygon(self) -> None:
        """
        Generates and appends a new layer of points to the fractal based on the most recently added points and polygon corners.
        """

        new_points = []
        for p1 in self._last_new:
            for p2 in self.corners:
                new_points.append(Point.mean([p1, p2]))
        self._points += new_points
        self._last_new = new_points.copy()


class MandelBrot(Fractal):
    def __init__(self, size: int, depth: int) -> None:
        """
        Initializes the fractal point collection.
        Args:
            size (int): The number of points to generate.
            depth (int): The maximum number of iterations for checking if a point belongs to the fractal set.
        """

        self._depth = depth
        self._points = []

        while len(self._points) < size:
            c = (
                random.uniform(-2, 0.5) + 1j * random.uniform(-1.1, 1.1)
            )  # Magic numbers are bounds, efficiency depends on how close to true shape of set.
            if self._check_sequence(c):
                self._points.append(Point(c.real, c.imag))

    def _check_sequence(self, c) -> bool:
        """
        Determines whether the sequence defined by the iteration z = z^2 + c remains bounded within a given depth.
        Args:
            c (complex): The complex parameter for the sequence.
        Returns:
            bool: True if the sequence does not escape (|z| <= 2) for all iterations up to self._depth, False otherwise.
        """

        z = 0 + 0j
        for _i in range(self._depth):
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
