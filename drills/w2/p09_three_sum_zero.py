def three_sum_zero(a):
    a = sorted(a)
    out = []
    for i in range(len(a) - 2):
        if i and a[i] == a[i - 1]:
            continue
        j, k = i + 1, len(a) - 1
        while j < k:
            s = a[i] + a[j] + a[k]
            if s == 0:
                out.append((a[i], a[j], a[k]))
                j += 1
                while j < k and a[j] == a[j - 1]:
                    j += 1
            elif s < 0:
                j += 1
            else:
                k -= 1
    return out
