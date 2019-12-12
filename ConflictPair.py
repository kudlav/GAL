"""
@author: Vladan Kudlac
"""
from typing import List, Tuple


class ConflictPair:

	def __init__(self, left: List[Tuple[int, int]] = [None, None], right: List[Tuple[int, int]] = [None, None]):
		self.l = left
		self.r = right
