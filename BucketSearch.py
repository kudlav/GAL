"""
@author: Vladan Kudlac
@see: https://www.geeksforgeeks.org/bucket-sort-2/
"""
from typing import List, Dict, Tuple


def bucket_sort(adj: List[int], v: int, nesting_depth: Dict[Tuple[int, int], int], size: int) -> List[int]:
	# Create size empty buckets
	buckets = []
	for i in range(size):
		buckets.append([])

	# Insert adj[i] into bucket[size*value]
	for w in adj:
		buckets[nesting_depth[(v, w)]].append(w)

	# Sort individual buckets using insertion sort.
	for bi in range(size):
		bucket = buckets[bi]
		for i in range(1, len(bucket)):
			up = bucket[i]
			j = i - 1
			while j >= 0 and bucket[j] > up:
				bucket[j + 1] = bucket[j]
				j -= 1
			bucket[j + 1] = up
		buckets[bi] = bucket

	# Concatenate all sorted buckets.
	index = 0
	for i in range(size):
		for j in range(len(buckets[i])):
			adj[index] = buckets[i][j]
			index += 1

	return adj
