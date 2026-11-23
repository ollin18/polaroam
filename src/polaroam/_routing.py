"""numba kernels for map matching.

- A bounded many-to-many Dijkstra: distances and paths between needed node pairs only. Each
  source searches until all its targets are settled or its distance limit is passed, and only
  the nodes it touched are reset, so the cost grows with the explored neighbourhood, not with
  the size of the network. Sources run in parallel.
- Viterbi over candidate positions, one trajectory per task, all in parallel.
"""

from __future__ import annotations

import numpy as np

try:
    import numba
    from numba import njit, prange
except ImportError:  # pragma: no cover - reported when map matching is used
    numba = None


def csr(n: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray):
    """CSR adjacency (indptr, indices, weights) of a directed graph with ``n`` nodes."""
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst[order].astype(np.int64), w[order].astype(np.float64)


if numba is not None:

    @njit(cache=True, inline="always")
    def _push(hd, hn, size, d, node):
        if size == hd.shape[0]:
            nd = np.empty(hd.shape[0] * 2, np.float64)
            nn = np.empty(hn.shape[0] * 2, np.int64)
            nd[:size] = hd[:size]
            nn[:size] = hn[:size]
            hd, hn = nd, nn
        i = size
        hd[i] = d
        hn[i] = node
        while i > 0:
            p = (i - 1) // 2
            if hd[p] <= hd[i]:
                break
            hd[p], hd[i] = hd[i], hd[p]
            hn[p], hn[i] = hn[i], hn[p]
            i = p
        return hd, hn, size + 1

    @njit(cache=True, inline="always")
    def _pop(hd, hn, size):
        d, node = hd[0], hn[0]
        size -= 1
        hd[0], hn[0] = hd[size], hn[size]
        i = 0
        while True:
            left, right, m = 2 * i + 1, 2 * i + 2, i
            if left < size and hd[left] < hd[m]:
                m = left
            if right < size and hd[right] < hd[m]:
                m = right
            if m == i:
                break
            hd[m], hd[i] = hd[i], hd[m]
            hn[m], hn[i] = hn[i], hn[m]
            i = m
        return d, node, size

    @njit(parallel=True, cache=True)
    def _search(indptr, indices, weights, n, sources, limits, tptr, targets, out, mode, path_ptr, path_out,
                n_blocks):
        """mode 0: distances into ``out``; 1: hop counts into ``out``; 2: also write the paths
        (target back to source) into ``path_out`` at ``path_ptr``. Sources are split into
        ``n_blocks`` interleaved blocks, each with its own workspace."""
        dist = np.full((n_blocks, n), np.inf)
        pred = np.full((n_blocks, n), -1, np.int64)
        stamp = np.zeros((n_blocks, n), np.int64)  # target of the current source (by source index + 1)
        settled = np.zeros((n_blocks, n), np.int64)
        for tid in prange(n_blocks):
            for k in range(tid, sources.shape[0], n_blocks):
                _one_source(indptr, indices, weights, sources, limits, tptr, targets, out, mode, path_ptr,
                            path_out, dist[tid], pred[tid], stamp[tid], settled[tid], k)

    @njit(cache=True)
    def _one_source(indptr, indices, weights, sources, limits, tptr, targets, out, mode, path_ptr, path_out,
                    dist, pred, stamp, settled, k):
        s, lim = sources[k], limits[k]
        mark = k + 1
        remaining = 0
        for j in range(tptr[k], tptr[k + 1]):
            t = targets[j]
            if stamp[t] != mark:
                stamp[t] = mark
                remaining += 1
        hd = np.empty(64, np.float64)
        hn = np.empty(64, np.int64)
        touched = np.empty(64, np.int64)
        n_touched = 0
        dist[s] = 0.0
        pred[s] = s
        touched[0] = s
        n_touched = 1
        hd, hn, size = _push(hd, hn, 0, 0.0, s)
        while size > 0 and remaining > 0:
            d, u, size = _pop(hd, hn, size)
            if d > lim:
                break
            if settled[u] == mark or d > dist[u]:
                continue
            settled[u] = mark
            if stamp[u] == mark:
                remaining -= 1
            for e in range(indptr[u], indptr[u + 1]):
                v = indices[e]
                nd = d + weights[e]
                if nd < dist[v] and nd <= lim:
                    if dist[v] == np.inf:
                        if n_touched == touched.shape[0]:
                            tt = np.empty(touched.shape[0] * 2, np.int64)
                            tt[:n_touched] = touched[:n_touched]
                            touched = tt
                        touched[n_touched] = v
                        n_touched += 1
                    dist[v] = nd
                    pred[v] = u
                    hd, hn, size = _push(hd, hn, size, nd, v)
        for j in range(tptr[k], tptr[k + 1]):
            t = targets[j]
            found = settled[t] == mark
            if mode == 0:
                out[j] = dist[t] if found else np.inf
            else:
                hops = -1
                if found:
                    hops = 0
                    x = t
                    while x != s:
                        if mode == 2:
                            path_out[path_ptr[j] + hops] = x
                        x = pred[x]
                        hops += 1
                    if mode == 2:
                        path_out[path_ptr[j] + hops] = s
                out[j] = hops
        for i in range(n_touched):
            v = touched[i]
            dist[v] = np.inf
            pred[v] = -1


def node_distances(graph, src: np.ndarray, dst: np.ndarray, lim: np.ndarray, paths: bool = False):
    """Shortest distances for node pairs (``inf`` when farther than ``lim``).

    With ``paths``, also each pair's node sequence as ``(pair, hop, node)`` arrays, hop 0
    being the target and the last hop the source.
    """
    if numba is None:
        raise ImportError("Map matching needs numba: pip install 'polaroam[roads]'.")
    indptr, indices, weights, n = graph
    order = np.lexsort((dst, src))
    s_sorted, t_sorted = src[order], dst[order]
    sources, start = np.unique(s_sorted, return_index=True)
    tptr = np.append(start, len(s_sorted)).astype(np.int64)
    limits = np.maximum.reduceat(lim[order], start) if len(start) else np.empty(0)
    dummy = np.empty(0, np.int64)
    nb = numba.get_num_threads()
    if not paths:
        out = np.empty(len(order), np.float64)
        _search(indptr, indices, weights, n, sources.astype(np.int64), limits.astype(np.float64), tptr,
                t_sorted.astype(np.int64), out, 0, dummy, dummy, nb)
        d = np.empty(len(order))
        d[order] = out
        return d
    hops = np.empty(len(order), np.float64)
    _search(indptr, indices, weights, n, sources.astype(np.int64), limits.astype(np.float64), tptr,
            t_sorted.astype(np.int64), hops, 1, dummy, dummy, nb)
    ok = hops >= 0
    counts = np.where(ok, hops + 1, 0).astype(np.int64)
    path_ptr = np.zeros(len(order), np.int64)
    np.cumsum(counts[:-1], out=path_ptr[1:])
    path_out = np.empty(int(counts.sum()), np.int64)
    dist = np.empty(len(order), np.float64)
    _search(indptr, indices, weights, n, sources.astype(np.int64), limits.astype(np.float64), tptr,
            t_sorted.astype(np.int64), dist, 2, path_ptr, path_out, nb)
    pair = np.repeat(order, counts)
    hop = np.arange(len(path_out)) - np.repeat(path_ptr, counts)
    return pair, hop, path_out


if numba is not None:

    @njit(parallel=True, cache=True)
    def _viterbi(key_ptr, step, emit, tptr, t_from, t_w, score, back, chosen, brk):
        """Per trajectory (rows ``key_ptr[k]:key_ptr[k+1]``, sorted by step then candidate):
        best candidate sequence. ``tptr``/``t_from``/``t_w``: transitions into each row (CSR by
        target row, sorted by source row). Writes ``chosen`` (1 for the chosen row of each
        step) and ``brk`` (1 where a matched part starts)."""
        for k in prange(key_ptr.shape[0] - 1):
            r0, r1 = key_ptr[k], key_ptr[k + 1]
            n_steps = 1
            for r in range(r0 + 1, r1):
                if step[r] != step[r - 1]:
                    n_steps += 1
            bstart = np.empty(n_steps + 1, np.int64)
            sbrk = np.zeros(n_steps, np.int8)
            j = 0
            bstart[0] = r0
            for r in range(r0 + 1, r1):
                if step[r] != step[r - 1]:
                    j += 1
                    bstart[j] = r
            bstart[n_steps] = r1
            for r in range(bstart[0], bstart[1]):
                score[r] = emit[r]
                back[r] = -1
            sbrk[0] = 1
            for j in range(1, n_steps):
                reach = False
                for r in range(bstart[j], bstart[j + 1]):
                    best = -np.inf
                    bk = -1
                    for t in range(tptr[r], tptr[r + 1]):
                        v = score[t_from[t]] + t_w[t]
                        if v > best:
                            best = v
                            bk = t_from[t]
                    if bk >= 0:
                        score[r] = best + emit[r]
                        back[r] = bk
                        reach = True
                    else:
                        score[r] = -np.inf
                        back[r] = -1
                if not reach:  # no possible transition: a new matched part starts here
                    sbrk[j] = 1
                    for r in range(bstart[j], bstart[j + 1]):
                        score[r] = emit[r]
                        back[r] = -1
            cur = -1
            for j in range(n_steps - 1, -1, -1):
                # the last step, or the step before a new part: the best candidate of the step
                if cur < 0 or (j + 1 < n_steps and sbrk[j + 1] == 1):
                    best = -np.inf
                    for r in range(bstart[j], bstart[j + 1]):
                        if score[r] > best:
                            best = score[r]
                            cur = r
                else:
                    cur = back[cur]
                chosen[cur] = 1
                brk[cur] = sbrk[j]


def viterbi(key_ptr, step, emit, tptr, t_from, t_w):
    """Run :func:`_viterbi`; returns ``(chosen, brk, back)`` per candidate row."""
    if numba is None:
        raise ImportError("Map matching needs numba: pip install 'polaroam[roads]'.")
    n = len(step)
    score = np.empty(n, np.float64)
    back = np.empty(n, np.int64)
    chosen = np.zeros(n, np.int8)
    brk = np.zeros(n, np.int8)
    _viterbi(key_ptr, step, emit, tptr, t_from, t_w, score, back, chosen, brk)
    return chosen.astype(bool), brk.astype(bool), back


if numba is not None:

    @njit(cache=True, inline="always")
    def _count(edges, counts, lo, hi, e):
        """``counts`` of edge ``e`` in the sorted slice ``edges[lo:hi]`` (0 if absent)."""
        if hi <= lo:
            return 0.0
        i = lo + np.searchsorted(edges[lo:hi], e)
        if i < hi and edges[i] == e:
            return counts[i]
        return 0.0

    @njit(parallel=True, cache=True)
    def _search_pref(indptr, indices, weights, arc_edge, n, sources, limits, tptr, targets, rg, tg, rptr, r_edges,
                     r_count, optr, o_edges, o_count, others, pref, out_cost, out_hops, mode, path_ptr, path_out,
                     n_blocks):
        """Like :func:`_search`, with arc weights lowered on the edges a group has used:
        ``weight * (1 - pref * share)``, where ``share`` is the edge's count in the group's
        route (``rg``: routine) minus its count in the trip itself (``tg``), over ``others``
        trips. Writes the cost and the hop count (mode 1), or also the paths (mode 2)."""
        dist = np.full((n_blocks, n), np.inf)
        pred = np.full((n_blocks, n), -1, np.int64)
        stamp = np.zeros((n_blocks, n), np.int64)
        settled = np.zeros((n_blocks, n), np.int64)
        for tid in prange(n_blocks):
            for k in range(tid, sources.shape[0], n_blocks):
                _one_pref(indptr, indices, weights, arc_edge, sources, limits, tptr, targets, rg, tg, rptr, r_edges,
                          r_count, optr, o_edges, o_count, others, pref, out_cost, out_hops, mode, path_ptr,
                          path_out, dist[tid], pred[tid], stamp[tid], settled[tid], k)

    @njit(cache=True)
    def _one_pref(indptr, indices, weights, arc_edge, sources, limits, tptr, targets, rg, tg, rptr, r_edges,
                  r_count, optr, o_edges, o_count, others, pref, out_cost, out_hops, mode, path_ptr, path_out,
                  dist, pred, stamp, settled, k):
        s, lim = sources[k], limits[k]
        r, t = rg[k], tg[k]
        n_other = others[k]
        mark = k + 1
        remaining = 0
        for j in range(tptr[k], tptr[k + 1]):
            x = targets[j]
            if stamp[x] != mark:
                stamp[x] = mark
                remaining += 1
        hd = np.empty(64, np.float64)
        hn = np.empty(64, np.int64)
        touched = np.empty(64, np.int64)
        dist[s] = 0.0
        pred[s] = s
        touched[0] = s
        n_touched = 1
        hd, hn, size = _push(hd, hn, 0, 0.0, s)
        while size > 0 and remaining > 0:
            d, u, size = _pop(hd, hn, size)
            if d > lim:
                break
            if settled[u] == mark or d > dist[u]:
                continue
            settled[u] = mark
            if stamp[u] == mark:
                remaining -= 1
            for e in range(indptr[u], indptr[u + 1]):
                v = indices[e]
                w = weights[e]
                if n_other > 0 and pref > 0:
                    ed = arc_edge[e]
                    c = _count(r_edges, r_count, rptr[r], rptr[r + 1], ed) - \
                        _count(o_edges, o_count, optr[t], optr[t + 1], ed)
                    if c > 0:
                        w = w * (1.0 - pref * min(1.0, c / n_other))
                nd = d + w
                if nd < dist[v] and nd <= lim:
                    if dist[v] == np.inf:
                        if n_touched == touched.shape[0]:
                            tt = np.empty(touched.shape[0] * 2, np.int64)
                            tt[:n_touched] = touched[:n_touched]
                            touched = tt
                        touched[n_touched] = v
                        n_touched += 1
                    dist[v] = nd
                    pred[v] = u
                    hd, hn, size = _push(hd, hn, size, nd, v)
        for j in range(tptr[k], tptr[k + 1]):
            x = targets[j]
            found = settled[x] == mark
            out_cost[j] = dist[x] if found else np.inf
            hops = -1
            if found:
                hops = 0
                y = x
                while y != s:
                    if mode == 2:
                        path_out[path_ptr[j] + hops] = y
                    y = pred[y]
                    hops += 1
                if mode == 2:
                    path_out[path_ptr[j] + hops] = s
            out_hops[j] = hops
        for i in range(n_touched):
            v = touched[i]
            dist[v] = np.inf
            pred[v] = -1


def preferred_paths(graph, arc_edge, src, dst, lim, rg, tg, others, routes, own, pref: float):
    """Shortest paths for node pairs, preferring edges a group has used before.

    Pair ``i`` searches from ``src[i]`` to ``dst[i]`` (cost limit ``lim[i]``) with the arc
    weights of routine ``rg[i]`` lowered by ``pref`` x the share of ``others[i]`` other trips
    that used each edge; ``routes`` / ``own`` are ``(ptr, edges, counts)`` CSR tables of edge
    counts per routine and per trip (``tg[i]``), each slice sorted by edge.

    Returns ``(cost, pair, hop, node)``: the cost per pair (``inf`` if not reached) and the
    node sequence of each found path (hop 0 is the target).
    """
    if numba is None:
        raise ImportError("Map matching needs numba: pip install 'polaroam[roads]'.")
    indptr, indices, weights, n = graph
    # one search per (pair group, source): pairs that share both search the same graph
    order = np.lexsort((dst, src, tg, rg))
    key = np.stack([rg[order], tg[order], src[order]], axis=1)
    new = np.ones(len(order), bool)
    if len(order) > 1:
        new[1:] = (key[1:] != key[:-1]).any(axis=1)
    start = np.flatnonzero(new)
    tptr = np.append(start, len(order)).astype(np.int64)
    sources = src[order][start].astype(np.int64)
    limits = np.maximum.reduceat(lim[order], start) if len(start) else np.empty(0)
    args = (indptr, indices, weights, arc_edge.astype(np.int64), n, sources, limits.astype(np.float64), tptr,
            dst[order].astype(np.int64), rg[order][start].astype(np.int64), tg[order][start].astype(np.int64),
            routes[0], routes[1], routes[2], own[0], own[1], own[2], others[order][start].astype(np.float64),
            float(pref))
    nb = numba.get_num_threads()
    cost = np.empty(len(order), np.float64)
    hops = np.empty(len(order), np.int64)
    dummy = np.empty(0, np.int64)
    _search_pref(*args, cost, hops, 1, dummy, dummy, nb)
    counts = np.where(hops >= 0, hops + 1, 0).astype(np.int64)
    path_ptr = np.zeros(len(order), np.int64)
    if len(order):
        np.cumsum(counts[:-1], out=path_ptr[1:])
    path_out = np.empty(int(counts.sum()), np.int64)
    _search_pref(*args, cost, hops, 2, path_ptr, path_out, nb)
    out = np.empty(len(order))
    out[order] = cost
    pair = np.repeat(order, counts)
    hop = np.arange(len(path_out)) - np.repeat(path_ptr, counts)
    return out, pair, hop, path_out
