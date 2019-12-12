"""
@author: Vladan Kudlac
"""
from typing import Tuple, List


class ConflictPair:

	l: List[Tuple[int, int]]
	r: List[Tuple[int, int]]

	def __init__(self, left: List[Tuple[int, int]] = None, right: List[Tuple[int, int]] = None):
		if left is None:
			left = [None, None]
		if right is None:
			right = [None, None]
		self.l = left
		self.r = right

	def swap(self) -> None:
		tmp = self.l
		self.l = self.r
		self.r = tmp
