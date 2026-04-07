"""
Base-(k+1) Graph — wrapper around the reference implementation by
Takezawa et al. (NeurIPS 2023).

Reference: https://github.com/yukiTakezawa/BaseGraph

This module provides a numpy-only interface that returns mixing
matrices as numpy arrays, suitable for our DFL benchmark.
"""

import numpy as np
import math
import copy


# ============================================================
# HyperHyperCube (Algorithm 1)
# ============================================================

class HyperHyperCube:
    """k-peer Hyper-hypercube Graph."""

    def __init__(self, n_nodes, max_degree=1):
        self.max_degree = max_degree
        self.n_nodes = n_nodes

        if n_nodes == 1:
            self.w_list = [np.eye(1)]
        else:
            # Check all prime factors ≤ max_degree + 1
            factors_list = self._split_node(n_nodes)
            self.w_list = self._construct(list(range(n_nodes)),
                                          factors_list, n_nodes)

    def _construct(self, node_list, factors_list, n_nodes):
        w_list = []
        for k_idx in range(len(factors_list)):
            w = np.zeros((n_nodes, n_nodes))
            b = np.zeros(n_nodes)

            for i_idx in range(len(node_list)):
                for nk in range(1, factors_list[k_idx]):
                    i = node_list[i_idx]
                    j = int(i + np.prod(factors_list[:k_idx]) * nk) % n_nodes

                    if b[i] < factors_list[k_idx] - 1 and \
                       b[j] < factors_list[k_idx] - 1:
                        b[i] += 1
                        b[j] += 1
                        w[i, j] = 1.0 / factors_list[k_idx]
                        w[j, i] = 1.0 / factors_list[k_idx]
                        w[i, i] = 1.0 / factors_list[k_idx]
                        w[j, j] = 1.0 / factors_list[k_idx]

            w_list.append(w)
        return w_list

    def _split_node(self, n_nodes):
        factors_list = []
        rest = n_nodes
        for factor in reversed(range(2, self.max_degree + 2)):
            while rest % factor == 0:
                factors_list.append(factor)
                rest = int(rest / factor)
                if rest == 1:
                    break
        factors_list.reverse()
        return factors_list


# ============================================================
# Simple Base-(k+1) Graph (Algorithm 2)
# ============================================================

class SimpleBaseGraph:
    """Simple Base-(k+1) Graph."""

    def __init__(self, n_nodes, max_degree=1, inner_edges=True):
        self.inner_edges = inner_edges
        self.max_degree = max_degree
        self.n_nodes = n_nodes
        self.w_list = self._construct()

    def _construct(self):
        node_list_list, n_nodes_list = self._split_nodes()
        node_list_list_list = self._split_nodes2(node_list_list)
        L = len(node_list_list)

        if self.n_nodes == 1:
            return [np.eye(1)]

        # Check if all prime factors ≤ max_degree + 1
        if self._max_prime_factor(self.n_nodes) <= self.max_degree + 1:
            return HyperHyperCube(self.n_nodes,
                                  max_degree=self.max_degree).w_list

        # Construct k-peer HyperHyperCubes
        hyperhyper_cubes = [
            HyperHyperCube(len(node_list_list[i]),
                           max_degree=self.max_degree)
            for i in range(L)
        ]
        hyperhyper_cubes2 = [
            HyperHyperCube(len(node_list_list_list[i][0]),
                           max_degree=self.max_degree)
            for i in range(L)
        ]
        max_length_of_hyper = len(hyperhyper_cubes[0].w_list)

        b = np.zeros(L)
        true_b = np.array([len(hc.w_list) for hc in hyperhyper_cubes2])

        w_list = []
        m = -1
        while True:
            m += 1
            w = np.zeros((self.n_nodes, self.n_nodes))
            isolated_nodes = None
            all_isolated_nodes = None

            for l in reversed(range(L)):
                if m < max_length_of_hyper:
                    length = len(hyperhyper_cubes[l].w_list)
                    w += self._extend(hyperhyper_cubes[l].w_list[m % length],
                                      node_list_list[l])

                elif m < max_length_of_hyper + l:
                    if isolated_nodes is None:
                        isolated_nodes = copy.deepcopy(
                            node_list_list_list[m - max_length_of_hyper])
                        all_isolated_nodes = [
                            node for nodes in isolated_nodes for node in nodes
                        ]

                    for i in node_list_list[l]:
                        a_l = len(isolated_nodes)
                        for kk in range(a_l):
                            j = isolated_nodes[kk].pop(-1)
                            all_isolated_nodes.remove(j)
                            weight = (n_nodes_list[m - max_length_of_hyper] /
                                      sum(n_nodes_list[m - max_length_of_hyper:]) /
                                      a_l)
                            w[i, j] = weight
                            w[j, i] = weight
                            w[j, j] = 1 - weight
                        w[i, i] = (1 - n_nodes_list[m - max_length_of_hyper] /
                                   sum(n_nodes_list[m - max_length_of_hyper:]))

                elif m == max_length_of_hyper + l and l != L - 1:
                    while (all_isolated_nodes is not None and
                           len(all_isolated_nodes) > 1 and self.inner_edges):
                        sampled = all_isolated_nodes[
                            :min(self.max_degree + 1, len(all_isolated_nodes))]
                        for nid in sampled:
                            all_isolated_nodes.remove(nid)
                        for ii in sampled:
                            for jj in sampled:
                                w[ii, jj] = 1.0 / len(sampled)

                else:
                    if n_nodes_list[l] < self.max_degree + 1:
                        length = len(hyperhyper_cubes[l].w_list)
                        w += self._extend(
                            hyperhyper_cubes[l].w_list[int(b[l] % length)],
                            node_list_list[l])
                    else:
                        a_l = len(node_list_list_list[l])
                        for kk in range(a_l):
                            length = len(hyperhyper_cubes2[l].w_list)
                            w += self._extend(
                                hyperhyper_cubes2[l].w_list[int(b[l] % length)],
                                node_list_list_list[l][kk])
                    b[l] += 1

            # Add self-loops
            for i in range(self.n_nodes):
                if w[i, i] == 0:
                    w[i, i] = 1.0
            w_list.append(w)

            if b[0] == len(hyperhyper_cubes2[0].w_list):
                break

        return w_list

    def _extend(self, w, node_list):
        new_w = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                new_w[node_list[i], node_list[j]] = w[i, j]
        return new_w

    def _split_nodes(self):
        factor = ((self.max_degree + 1) **
                  int(math.log(self.n_nodes, self.max_degree + 1)))
        n_nodes_list = []

        while sum(n_nodes_list) != self.n_nodes:
            rest = self.n_nodes - sum(n_nodes_list)
            if rest >= factor:
                n_nodes_list.append((rest // factor) * factor)
            factor = int(factor / (self.max_degree + 1))

        node_list = list(range(self.n_nodes))
        node_list_list = []
        for i in range(len(n_nodes_list)):
            start = sum(n_nodes_list[:i])
            end = start + n_nodes_list[i]
            node_list_list.append(node_list[start:end])

        return node_list_list, n_nodes_list

    def _split_nodes2(self, node_list_list):
        result = []
        for node_list in node_list_list:
            n_nodes = len(node_list)
            power = math.gcd(
                n_nodes,
                (self.max_degree + 1) **
                int(math.log(n_nodes, self.max_degree + 1)))
            rest = int(n_nodes / power)
            chunks = []
            for i in range(rest):
                chunks.append(node_list[i * power:(i + 1) * power])
            result.append(chunks)
        return result

    @staticmethod
    def _max_prime_factor(n):
        """Find the largest prime factor of n."""
        factor = 2
        max_f = 1
        while factor * factor <= n:
            while n % factor == 0:
                max_f = factor
                n //= factor
            factor += 1
        if n > 1:
            max_f = n
        return max_f


# ============================================================
# Base-(k+1) Graph (Algorithm 3)
# ============================================================

class BaseGraphConstructor:
    """Full Base-(k+1) Graph construction."""

    def __init__(self, n_nodes, max_degree=1, inner_edges=True):
        self.max_degree = max_degree
        self.n_nodes = n_nodes
        self.inner_edges = inner_edges
        self.w_list = self._construct()

    def _construct(self):
        node_list_list1, node_list_list2, n_power, n_rest = \
            self._split_nodes()

        simple_adics = [
            SimpleBaseGraph(len(node_list_list1[i]),
                            max_degree=self.max_degree,
                            inner_edges=self.inner_edges)
            for i in range(n_power)
        ]
        hyper_cubes = [
            HyperHyperCube(len(node_list_list2[i]),
                           max_degree=self.max_degree)
            for i in range(n_rest)
        ]

        # Check: is Simple Base graph on full n shorter?
        g = SimpleBaseGraph(self.n_nodes, max_degree=self.max_degree,
                            inner_edges=self.inner_edges)
        if len(g.w_list) < len(simple_adics[0].w_list) + \
                            len(hyper_cubes[0].w_list):
            return g.w_list

        w_list = []

        # Phase 1: Simple Base-(k+1) within each V_l
        for m in range(len(simple_adics[0].w_list)):
            w = np.zeros((self.n_nodes, self.n_nodes))
            for l in range(n_power):
                w += self._extend(simple_adics[l].w_list[m],
                                  node_list_list1[l])
            w_list.append(w)

        # Phase 2: HyperHyperCube across U_j groups
        for m in range(len(hyper_cubes[0].w_list)):
            w = np.zeros((self.n_nodes, self.n_nodes))
            for l in range(n_rest):
                w += self._extend(hyper_cubes[l].w_list[m],
                                  node_list_list2[l])
            w_list.append(w)

        return w_list

    def _extend(self, w, node_list):
        new_w = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                new_w[node_list[i], node_list[j]] = w[i, j]
        return new_w

    def _split_nodes(self):
        factors = [
            n ** int(math.log(self.n_nodes, n))
            for n in range(2, self.max_degree + 2)
        ]
        factor = int(np.prod(factors))
        n_power = math.gcd(self.n_nodes, factor)
        n_rest = int(self.n_nodes / n_power)

        node_list = list(range(self.n_nodes))
        node_list_list1 = []
        for i in range(n_power):
            node_list_list1.append(
                node_list[n_rest * i:n_rest * (i + 1)])

        node_list_list2 = [[] for _ in range(n_rest)]
        for i in range(n_power):
            for j in range(n_rest):
                node_list_list2[j].append(node_list_list1[i][j])

        return node_list_list1, node_list_list2, n_power, n_rest


# ============================================================
# Public API
# ============================================================

def build_base_graph(n, k):
    """
    Build Base-(k+1) Graph.

    Args:
        n: number of nodes
        k: maximum degree per round

    Returns:
        List of numpy mixing matrices (doubly stochastic), one per round.
    """
    bg = BaseGraphConstructor(n, max_degree=k, inner_edges=True)
    return bg.w_list


def verify_finite_time_consensus(W_list, tol=1e-10):
    """Verify product of W matrices = (1/n)·11^T."""
    n = W_list[0].shape[0]
    product = np.eye(n)
    for W in W_list:
        product = W @ product
    target = np.ones((n, n)) / n
    error = np.max(np.abs(product - target))
    return error < tol, error


if __name__ == "__main__":
    print("Testing Base-(k+1) Graph (reference implementation)")
    print("=" * 55)

    for n in [4, 5, 6, 7, 8, 9, 10, 12, 16, 25, 32, 64]:
        for k in [1, 2, 4]:
            if k >= n:
                continue
            W_list = build_base_graph(n, k)
            ok, err = verify_finite_time_consensus(W_list)
            status = "OK" if ok else f"FAIL (err={err:.2e})"
            max_deg = 0
            for W in W_list:
                degs = (np.abs(W) > 1e-12).sum(axis=1) - 1
                max_deg = max(max_deg, int(degs.max()))
            print(f"  n={n:3d} k={k}: {len(W_list):3d} rounds, "
                  f"max_deg={max_deg}, {status}")
