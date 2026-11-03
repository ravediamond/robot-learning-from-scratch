def dedupe_sorted(a):
    if not a:
        return 0
    k = 1
    for x in a[1:]:
        if x != a[k - 1]:
            a[k] = x
            k += 1
    return k
