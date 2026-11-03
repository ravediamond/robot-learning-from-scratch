# Pattern notes (read only after an attempt)

**Hash map (p01).** "Have I seen the complement before?" One pass, store value → index,
check `target - x` before inserting x. O(n) time, O(n) space. Edge: duplicates, the pair
using the same element twice.

**Stack (p02).** Matching nested things: push openers, on a closer pop and compare. Valid
iff every pop matches and the stack is empty at the end. O(n).

**Monotonic deque (p03).** Keep indices whose values are decreasing. For each new x: pop
from the back while the back value ≤ x (they can never be the max again); pop from the
front if it left the window; the front is the max. Each index enters and leaves once: O(n).

**Merge intervals (p04).** Sort by start. Walk: if the next start ≤ current end (+1 if
touching counts), extend the end with max; else close the current one. O(n log n).

**Binary search on a predicate (p05).** Invariant: answer in [lo, hi]. lo = 0, hi = n.
While lo < hi: mid = (lo + hi) // 2; if bad(mid) hi = mid else lo = mid + 1. Return lo.
Works for any monotone yes/no question, not only sorted arrays. O(log n) calls.

## Weeks 2–8

**Two pointers (p06–p09).** Sorted input + a target → move the pointer that brings you closer. Sliding
window with a running sum for "longest/shortest contiguous" with non-negative values. 3-sum = sort + fix one + two pointers; skip duplicates.

**Heaps (p10–p13).** Top-k: keep a heap of size k (max-heap of the worst kept element). Merge k sorted:
heap of the heads. Running median: max-heap (lower half) + min-heap (upper half), rebalance sizes.

**Grid BFS/DFS (p14–p17).** Visited set + explicit stack/queue (no recursion on big grids). BFS gives
shortest paths in unweighted grids; multi-source BFS starts with all sources in the queue.

**Graphs (p18–p21).** Union-find for components (path compression + union by size). Kahn's algorithm
for topological order (in-degrees; cycle if not all nodes are output). Dijkstra = BFS with a priority
queue, non-negative weights, skip stale entries. Directed cycle = an edge to a node still on the DFS stack.

**DP (p22–p29).** Define the state in one sentence, write the recurrence, choose the iteration order,
then compress memory to one row. 0/1 knapsack iterates capacity BACKWARD; unbounded (coins) forward.
LIS in O(n log n): tails array + bisect. Interval scheduling: sort by END and greedily keep compatible ones.

**Design (p30–p33).** LRU = hash map + doubly linked list (OrderedDict). Trie = nested dicts with an
end marker. Ring buffer = fixed array + write index + running sum. In-place rotation = transpose + reverse rows.
