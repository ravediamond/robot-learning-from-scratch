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
