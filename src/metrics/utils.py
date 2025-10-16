def levenshtein(seq1, seq2):
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if seq1[i - 1] == seq2[j - 1] else 1)
            cur[j] = min(ins, dele, sub)
        prev = cur
    return prev[m]