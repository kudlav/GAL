"""
@author: Vladan Kudlac
"""
from typing import Tuple


class Interval:

	low: Tuple[int, int]
	high: Tuple[int, int]

	def __init__(self, low: Tuple[int, int] = None, high: Tuple[int, int] = None):
		self.low = low
		self.high = high

	def empty(self):
		return self.low is None and self.high is None
