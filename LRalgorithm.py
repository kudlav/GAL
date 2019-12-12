"""
@author: Vladan Kudlac
"""
from collections import defaultdict
from typing import Tuple, List, Dict
from ConflictPair import ConflictPair
import numpy


class LRPlanarityCheck:

	parent_edge: List[Tuple[int, int]]
	height: List[int]
	orientedGraph: List[Tuple[int, int]]
	lowpt: Dict[Tuple[int, int], int]
	nesting_depth: Dict[Tuple[int, int], int]

	adj_list: List[List[int]]

	graph: numpy.array
	lowpt2: Dict[Tuple[int, int], int]

	s: List[ConflictPair]
	stack_bottom: Dict[Tuple[int, int], ConflictPair]
	lowpt_edge: Dict[Tuple[int, int], Tuple[int, int]]
	ref: defaultdict  # Dict[Tuple[int, int], Tuple[int, int]]
	side: defaultdict  # Dict[Tuple[int, int], int]

	def __init__(self, graph: numpy.array):
		# Used in both DFS traversals
		self.parent_edge = [None] * graph[0].size
		self.height = [None] * graph[0].size
		self.orientedGraph = []
		self.lowpt = {}
		self.nesting_depth = {}

		self.adj_list = [None] * graph[0].size

		# Used in first DFS orientation, removed after that
		self.graph = graph
		self.lowpt2 = {}

		# Used in second DFS traversal
		self.s = []
		self.stack_bottom = {}
		self.lowpt_edge = {}
		self.ref = defaultdict(lambda: None)
		self.side = defaultdict(lambda: 1)

	def simple_check(self) -> bool:
		"""
		Euler's Relation planarity check
		@return: bool False when not planar, True when it still can be planar
		"""
		if self.graph.size == 0:
			return True

		v = self.graph[0].size
		e = self.graph.sum() / 2

		return v < 3 or e <= (3 * v - 6)

	def run(self) -> bool:
		"""
		Left-Right planarity algorithm
		@return: bool False when not planar, True when planar
		"""
		if self.graph.size == 0:
			return True

		roots = []

		# Orientation
		for s in range(self.graph[0].size):
			if self.height[s] is None:
				self.height[s] = 0
				roots.append(s)
				self.lr_orientation(s)
			# sort adjacency list according to non-decreasing nesting_depth O(n * log n)
			if self.adj_list[s] is not None:
				self.adj_list[s] = sorted(self.adj_list[s], key=lambda x: self.nesting_depth[(s, x)])
			else:
				self.adj_list[s] = []

		del self.graph
		del self.lowpt2

		# Testing
		for s in roots:
			if not self.lr_test(s):
				return False

		return True

	def lr_orientation(self, v: int) -> None:
		"""
		Phase 1 - DFS orientation and nesting order
		@param v: int vertex
		"""
		e = self.parent_edge[v]
		for w in range(self.graph[v].size):
			if self.graph[v][w] == 1 and not ((v, w) in self.orientedGraph or (w, v) in self.orientedGraph):
				if self.adj_list[v] is None:
					self.adj_list[v] = [w]
				else:
					self.adj_list[v].append(w)
				self.orientedGraph.append((v, w))
				self.lowpt[(v, w)] = self.height[v]
				self.lowpt2[(v, w)] = self.height[v]

				if self.height[w] is None:  # tree edge
					self.parent_edge[w] = (v, w)
					self.height[w] = self.height[v] + 1
					self.lr_orientation(w)
				else:  # back edge
					self.lowpt[(v, w)] = self.height[w]

				# determine nesting depth
				self.nesting_depth[(v, w)] = 2 * self.lowpt[(v, w)]
				if self.lowpt2[(v, w)] < self.height[v]:  # chordal
					self.nesting_depth[(v, w)] += 1

				# update lowpoints of parent edge e
				if e is not None:
					if self.lowpt[(v, w)] < self.lowpt[e]:
						self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[(v, w)])
						self.lowpt[e] = self.lowpt[(v, w)]
					elif self.lowpt[(v, w)] > self.lowpt[e]:
						self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[(v, w)])
					else:
						self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[(v, w)])

	def lr_test(self, v: int) -> bool:
		"""
		Phase 2 - Testing for LR partition
		@param v: int vertex
		@return: bool False when not planar, True when it still can be planar
		"""
		e = self.parent_edge[v]
		for w in self.adj_list[v]:
			ei = (v, w)
			self.stack_bottom[ei] = self.top_s()
			if ei == self.parent_edge[w]:  # tree edge
				if not self.lr_test(w):
					return False
			else:  # back edge
				self.lowpt_edge[ei] = ei
				self.s.append(ConflictPair(right=[ei, ei]))
			if self.lowpt[ei] < self.height[v]:  # ei has return edge
				if w == self.adj_list[v][0]:
					self.lowpt_edge[e] = self.lowpt_edge[ei]
				else:
					# Add constraints of ei
					if not self.add_constraints(ei, e):
						return False

		if e is not None:  # v is not root
			u = e[0]
			# Trim back edges ending at parent u
			self.trim_back_edges(u)
			# Side of e is side of a highest return edge
			if self.lowpt[e] < self.height[u]:  # e has return edge
				hl = self.top_s().l[1]
				hr = self.top_s().r[1]
				if hl is not None and (hr is None or self.lowpt[hl] > self.lowpt[hr]):
					self.ref[e] = hl
				else:
					self.ref[e] = hr

		return True

	def add_constraints(self, ei: Tuple[int, int], e: Tuple[int, int]) -> bool:
		"""
		Add constraints of ei (phase 2)
		@param e: Tuple[int, int]
		@param ei: Tuple[int, int]
		@return: bool False when not planar, True when it still can be planar
		"""
		p = ConflictPair()
		# merge return edges of ei into p.r
		while True:
			q = self.s.pop()
			if not (q.l[0] is None and q.l[1] is None):
				q.swap()
			if not (q.l[0] is None and q.l[1] is None):
				return False  # HALT: not planar
			else:
				if self.lowpt[q.r[0]] > self.lowpt[e]:  # merge intervals
					if p.r[0] is None and p.r[1] is None:
						p.r[1] = q.r[1]
					else:
						self.ref[p.r[0]] = q.r[1]
					p.r[0] = q.r[0]
				else:  # make consistent
					self.ref[q.r[0]] = self.lowpt_edge[e]
			if self.top_s() == self.stack_bottom[ei]:
				break

		# merge conflicting return edges of e(1), ... , e(i-1) into p.l
		while self.conflicting(self.top_s().l, ei) or self.conflicting(self.top_s().r, ei):
			q = self.s.pop()
			if self.conflicting(q.r, ei):
				q.swap()
			if self.conflicting(q.r, ei):
				return False  # HALT: not planar
			else:  # merge interval below lowpt[ei] into p.r
				self.ref[p.r[0]] = q.r[1]
				if q.r[0] is not None:
					p.r[0] = q.r[0]
			if p.l[0] is None and p.l[1] is None:
				p.l[1] = q.l[1]
			else:
				self.ref[p.l[0]] = q.l[1]
			p.l[0] = q.l[0]

		if not (p.l[0] is None and p.l[1] is None and p.r[0] is None and p.r[1] is None):
			self.s.append(p)

		return True

	def trim_back_edges(self, u: int):
		"""
		Remove back edges ending at parent u (phase 2)
		@param u:
		@return:
		"""
		# drop entire conflict pairs
		while len(self.s) and self.lowest(self.top_s()) == self.height[u]:
			p = self.s.pop()
			if p.l[0] is not None:
				self.side[p.l[0]] = -1

		# one more conflict pair to consider
		if len(self.s):
			p = self.s.pop()
			# trim left interval
			while p.l[1] is not None and p.l[1][1] == u:
				p.l[1] = self.ref[p.l[1]]
			if p.l[1] is None and p.l[0] is not None:  # just emptied
				self.ref[p.l[0]] = p.r[0]
				self.side[p.l[0]] = -1
				p.l = (None, p.l[1])
			# trim right interval
			while p.r[1] is not None and p.r[1][1] == u:
				p.r[1] = self.ref[p.r[1]]
			if p.r[1] is None and p.r[0] is not None:  # just emptied
				self.ref[p.r[0]] = p.l[0]
				self.side[p.r[0]] = -1
				p.r = (None, p.r[1])
			self.s.append(p)

	def conflicting(self, i: List[Tuple[int, int]], b: Tuple[int, int]) -> bool:
		"""
		Check conflicting edge against interval
		@param i: List[Tuple[int, int]] interval
		@param b: Tuple[int, int] edge
		@return: bool True when conflicting, otherwise False
		"""
		return not (i[0] is None and i[1] is None) and self.lowpt[i[1]] > self.lowpt[b]

	def top_s(self) -> ConflictPair:
		"""
		Return last item on stack or None when empty
		@return: ConflictPair | None
		"""
		return self.s[-1] if len(self.s) else None

	def lowest(self, p: ConflictPair) -> int:
		if p.l[0] is None and p.l[1] is None:
			return self.lowpt[p.r[0]]
		if p.r[0] is None and p.r[1] is None:
			return self.lowpt[p.l[0]]
		return min(self.lowpt[p.r[0]], self.lowpt[p.l[0]])


if __name__ == '__main__':

	print("K3 Triangle (planar)")
	graph = numpy.empty((3, 3))
	graph.fill(1)
	graph -= numpy.diag(graph.diagonal())
	lrTest = LRPlanarityCheck(graph)
	print("Simple check: ", lrTest.simple_check())
	print("LR test: ", lrTest.run())
	print(graph)

	print("Graph by Dylan Emery (planar)")
	graph = numpy.empty((9, 9))
	graph.fill(0)
	graph[0][1] = 1
	graph[0][3] = 1
	graph[0][8] = 1
	graph[2][1] = 1
	graph[2][3] = 1
	graph[2][5] = 1
	graph[4][3] = 1
	graph[4][5] = 1
	graph[4][8] = 1
	graph[6][1] = 1
	graph[6][5] = 1
	graph[7][5] = 1
	graph[7][8] = 1
	graph += graph.T - numpy.diag(graph.diagonal())
	lrTest = LRPlanarityCheck(graph)
	print("Simple check: ", lrTest.simple_check())
	print("LR test: ", lrTest.run())
	print(graph)

	print("\nK3,3 (not planar)")
	graph = numpy.empty((6, 6))
	graph.fill(0)
	graph[0][3] = 1
	graph[0][4] = 1
	graph[0][5] = 1
	graph[1][3] = 1
	graph[1][4] = 1
	graph[1][5] = 1
	graph[2][3] = 1
	graph[2][4] = 1
	graph[2][5] = 1
	graph += graph.T - numpy.diag(graph.diagonal())
	lrTest = LRPlanarityCheck(graph)
	print("Simple check: ", lrTest.simple_check())
	print("LR test: ", lrTest.run())
	print(graph)

	print("\nK5 (not planar)")
	graph = numpy.empty((5, 5))
	graph.fill(1)
	graph -= numpy.diag(graph.diagonal())
	lrTest = LRPlanarityCheck(graph)
	print("Simple check: ", lrTest.simple_check())
	print("LR test: ", lrTest.run())
	print(graph)

	print("\nK6 (not planar)")
	graph = numpy.empty((6, 6))
	graph.fill(1)
	graph -= numpy.diag(graph.diagonal())
	lrTest = LRPlanarityCheck(graph)
	print("Simple check: ", lrTest.simple_check())
	print("LR test: ", lrTest.run())
	print(graph)
