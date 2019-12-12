"""
@author: Vladan Kudlac
"""
from ConflictPair import ConflictPair
import numpy


class LRPlanarityCheck:

	def __init__(self, graph: numpy.array):
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
		self.stack_bottom = []
		self.ref = {}
		self.lowpt_edge = {}

	def simple_check(self) -> bool:
		if self.graph.size == 0:
			return True

		v = self.graph[0].size
		e = self.graph.sum() / 2

		return e <= (3 * v - 6)

	def run(self) -> bool:
		"""
		Left-Right planarity algorithm
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

	def lr_orientation(self, v: int):
		"""
		Phase 1 - DFS orientation and nesting order
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

	def lr_test(self, v: int):
		"""
		Phase 2 - Testing for LR partition
		"""
		e = self.parent_edge[v]
		for w in self.adj_list[v]:
			ei = (v, w)
			self.stack_bottom = top(self.s)
			if ei == self.parent_edge[w]:  # tree edge
				if not self.lr_test(w):
					return False
			else:  # back edge
				self.lowpt_edge[ei] = ei
				self.s.append(ConflictPair(right=[ei, ei]))
			if self.lowpt[ei] < self.height[v]:  # ei has return edge
				if ei == self.adj_list[v][0]:
					self.lowpt_edge[e] = self.lowpt_edge[ei]
				else:
					# Add constraints of ei
					if not self.add_constraints(ei):
						return False

		if e is not None:  # v is not root
			u = e[0]
			# Trim back edges ending at parent u
			self.trim_back_edges(u)
			# Side of e is side of a highest return edge
			if self.lowpt[e] < self.height[u]:  # e has return edge
				hl = top(self.s).l[1]
				hr = top(self.s).r[1]
				if hl is not None and (hr is not None or self.lowpt[hl] > self.lowpt[hr]):
					self.ref[e] = hl
				else:
					self.ref[e] = hr

	def add_constraints(self, ei: tuple) -> bool:
		return True

	def trim_back_edges(self, u: int):
		pass


def top(stack):
	"""
	Return last item on stack or null when empty
	@param stack: List
	@return:
	"""
	return stack[-1] if len(stack) != 0 else None


if __name__ == '__main__':
	graph = numpy.empty((10, 10))
	graph.fill(0)
	graph[0][5] = 1
	graph[5][6] = 1
	graph[6][0] = 1
	graph[6][7] = 1
	graph[7][5] = 1
	lrTest = LRPlanarityCheck(graph)
	print("Simple check: ", lrTest.simple_check())
	print("LR test: ", lrTest.run())
