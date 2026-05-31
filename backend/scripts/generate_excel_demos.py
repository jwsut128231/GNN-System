"""Generate demo .xlsx files matching the graph_data_template (V3 — 2026-04-28).

Produces files under backend/demo_data/:
    * demo_multigraph_homo.v2.xlsx          — 30 graphs, homogeneous, graph regression
    * demo_multigraph_homo_large.v2.xlsx    — 100 graphs, homogeneous, graph regression
    * demo_multigraph_hetero.v2.xlsx        — 30 graphs, 3 node types (cell/pin/net),
                                              2 edge types (cell_pin/pin_net),
                                              graph regression (total_wirelength)
    * demo_hetero_multifeature.v3.xlsx      — 30 graphs, string Graph_IDs (G001…G030),
                                              2 node types (CAP/RES), per-graph
                                              variable feature groups for CAP
                                              (X_1+X_2, X_1+X_3, or X_2+X_3),
                                              graph regression (target_y, predictable)

Run with:
    cd backend && python scripts/generate_excel_demos.py
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import pandas as pd


# ── Union-Find (for connectivity enforcement) ─────────────────────────────

class _UnionFind:
    def __init__(self, nodes: Iterable) -> None:
        self._parent: dict = {n: n for n in nodes}

    def find(self, x):
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a, b) -> None:
        self._parent[self.find(a)] = self.find(b)

    def components(self) -> dict:
        """Return {root: [members]} for every component."""
        groups: dict = {}
        for n in self._parent:
            r = self.find(n)
            groups.setdefault(r, []).append(n)
        return groups

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "demo_data"
OUT.mkdir(parents=True, exist_ok=True)


def _write(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


# ── Homogeneous demo (graph_regression) ───────────────────────────────────
# Schema: Parameter sheet declares Type="default" for all levels.
# Data sheets (Node / Edge / Graph) have NO Type column.

def _homo_edge(gid: int, s: int, d: int, rng: random.Random) -> dict:
    return {
        "Graph_ID": gid, "Source_Node_ID": s, "Target_Node_ID": d,
        "wire_cap_ff": round(rng.uniform(0.1, 5.0), 3),
        "wire_length_um": round(rng.uniform(1.0, 100.0), 2),
    }


def make_homo(n_graphs: int = 30) -> dict[str, pd.DataFrame]:
    rng = random.Random(SEED)
    parameter = pd.DataFrame([
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "delay_ps", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "area_um2", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "fanout", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "drive", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "default", "Parameter": "depth", "Weight": None},
        {"XY": "X", "Level": "Edge", "Type": "default", "Parameter": "wire_cap_ff", "Weight": None},
        {"XY": "X", "Level": "Edge", "Type": "default", "Parameter": "wire_length_um", "Weight": None},
        {"XY": "X", "Level": "Graph", "Type": "default", "Parameter": "num_cells", "Weight": None},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "target_delay", "Weight": 1.0},
    ])

    node_rows, edge_rows, graph_rows = [], [], []
    for gid in range(1, n_graphs + 1):
        n_nodes = rng.randint(20, 50)
        node_ids = list(range(n_nodes))
        total_delay = 0.0
        for nid in node_ids:
            delay = rng.randint(5, 60)
            total_delay += delay
            node_rows.append({
                "Graph_ID": gid, "Node": nid,
                "delay_ps": delay,
                "area_um2": round(rng.uniform(0.3, 3.0), 3),
                "fanout": rng.randint(1, 10),
                "drive": rng.choice([1, 2, 4, 8, 16]),
                "depth": rng.randint(1, 10),
            })
        n_edges = int(n_nodes * 1.5)
        g_edges: list[dict] = []
        for _ in range(n_edges):
            s = rng.choice(node_ids)
            d = rng.choice(node_ids)
            if s == d:
                continue
            g_edges.append(_homo_edge(gid, s, d, rng))

        # Coverage pass: ensure every node appears in at least one edge.
        covered = set()
        for e in g_edges:
            covered.add(e["Source_Node_ID"])
            covered.add(e["Target_Node_ID"])
        for nid in node_ids:
            if nid not in covered:
                # Connect isolated node to a random different node.
                others = [x for x in node_ids if x != nid]
                partner = rng.choice(others)
                g_edges.append(_homo_edge(gid, nid, partner, rng))

        # Connectivity pass: merge disconnected components into one.
        uf = _UnionFind(node_ids)
        for e in g_edges:
            uf.union(e["Source_Node_ID"], e["Target_Node_ID"])
        comps = uf.components()
        while len(comps) > 1:
            roots = list(comps.keys())
            # Pick the largest component as anchor.
            largest = max(roots, key=lambda r: len(comps[r]))
            anchor = rng.choice(comps[largest])
            # Pick a node from any other component.
            other_root = next(r for r in roots if r != largest)
            bridge = rng.choice(comps[other_root])
            g_edges.append(_homo_edge(gid, anchor, bridge, rng))
            uf.union(anchor, bridge)
            comps = uf.components()

        edge_rows.extend(g_edges)
        graph_rows.append({
            "Graph_ID": gid,
            "num_cells": n_nodes,
            "target_delay": round(total_delay / n_nodes + rng.uniform(-2, 2), 3),
        })

    return {
        "Parameter": parameter,
        "Node": pd.DataFrame(node_rows),
        "Edge": pd.DataFrame(edge_rows),
        "Graph": pd.DataFrame(graph_rows),
    }


# ── Heterogeneous demo (graph_regression) ────────────────────────────────
# 3 node types: cell, pin, net
# 2 edge types: cell_pin (cell→pin connections), pin_net (pin→net connections)
# Graph-level Y: total_wirelength (regression)
#
# Node sheet has a ``Type`` column; Edge sheet has a ``Type`` column.
# Parameter sheet declares features per node/edge type + graph-level Y.

def make_hetero(n_graphs: int = 30) -> dict[str, pd.DataFrame]:
    rng = random.Random(SEED)

    # Feature layout (includes two SHARED features to exercise Fix 2):
    #   area_um2     — declared for BOTH cell AND pin  (shared)
    #   cap_ff       — declared for BOTH pin  AND net  (shared)
    #   cell_delay_ps — cell only
    #   cell_drive    — cell only
    #   pin_direction — pin only
    #   net_fanout    — net only
    parameter = pd.DataFrame([
        # Cell node features
        {"XY": "X", "Level": "Node", "Type": "cell", "Parameter": "cell_delay_ps", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "cell", "Parameter": "area_um2", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "cell", "Parameter": "cell_drive", "Weight": None},
        # Pin node features (area_um2 shared with cell; cap_ff shared with net)
        {"XY": "X", "Level": "Node", "Type": "pin", "Parameter": "area_um2", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "pin", "Parameter": "cap_ff", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "pin", "Parameter": "pin_direction", "Weight": None},
        # Net node features (cap_ff shared with pin)
        {"XY": "X", "Level": "Node", "Type": "net", "Parameter": "cap_ff", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "net", "Parameter": "net_fanout", "Weight": None},
        # cell_pin edge features
        {"XY": "X", "Level": "Edge", "Type": "cell_pin", "Parameter": "cp_resistance_ohm", "Weight": None},
        # pin_net edge features
        {"XY": "X", "Level": "Edge", "Type": "pin_net", "Parameter": "pn_wire_length_um", "Weight": None},
        # Graph-level X + Y
        {"XY": "X", "Level": "Graph", "Type": "default", "Parameter": "num_cells", "Weight": None},
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "total_wirelength", "Weight": 1.0},
    ])

    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    graph_rows: list[dict] = []

    global_node_id = 0

    for gid in range(1, n_graphs + 1):
        n_cells = rng.randint(5, 15)
        n_pins_per_cell = rng.randint(2, 4)
        n_nets = rng.randint(3, 8)

        cell_ids = list(range(global_node_id, global_node_id + n_cells))
        global_node_id += n_cells
        pin_ids = list(range(global_node_id, global_node_id + n_cells * n_pins_per_cell))
        global_node_id += n_cells * n_pins_per_cell
        net_ids = list(range(global_node_id, global_node_id + n_nets))
        global_node_id += n_nets

        for cid in cell_ids:
            node_rows.append({
                "Graph_ID": gid, "Node": cid, "Type": "cell",
                # cell-private features
                "cell_delay_ps": rng.randint(5, 60),
                "cell_drive": rng.choice([1, 2, 4, 8]),
                # shared with pin
                "area_um2": round(rng.uniform(0.3, 3.0), 3),
                # NaN for non-cell columns
                "cap_ff": None, "pin_direction": None, "net_fanout": None,
            })

        for pid in pin_ids:
            node_rows.append({
                "Graph_ID": gid, "Node": pid, "Type": "pin",
                # shared with cell
                "area_um2": round(rng.uniform(0.05, 1.5), 3),
                # shared with net
                "cap_ff": round(rng.uniform(0.01, 0.5), 4),
                # pin-private
                "pin_direction": rng.choice([0, 1]),
                # NaN for non-pin columns
                "cell_delay_ps": None, "cell_drive": None, "net_fanout": None,
            })

        for nid in net_ids:
            node_rows.append({
                "Graph_ID": gid, "Node": nid, "Type": "net",
                # shared with pin
                "cap_ff": round(rng.uniform(0.1, 2.0), 3),
                # net-private
                "net_fanout": rng.randint(1, 8),
                # NaN for non-net columns
                "cell_delay_ps": None, "cell_drive": None,
                "area_um2": None, "pin_direction": None,
            })

        # cell_pin edges: each cell connects to its pins
        total_wl = 0.0
        g_edges: list[dict] = []
        for i, cid in enumerate(cell_ids):
            for j in range(n_pins_per_cell):
                pid = pin_ids[i * n_pins_per_cell + j]
                resistance = round(rng.uniform(1.0, 50.0), 2)
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": cid, "Target_Node_ID": pid,
                    "Type": "cell_pin",
                    "cp_resistance_ohm": resistance,
                    "pn_wire_length_um": None,
                })

        # pin_net edges: each net connects to random pins
        for nid in net_ids:
            connected = rng.sample(pin_ids, min(rng.randint(1, 3), len(pin_ids)))
            for pid in connected:
                wl = round(rng.uniform(5.0, 200.0), 2)
                total_wl += wl
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": pid, "Target_Node_ID": nid,
                    "Type": "pin_net",
                    "cp_resistance_ohm": None,
                    "pn_wire_length_um": wl,
                })

        # Coverage pass: ensure every node appears in at least one edge.
        covered = set()
        for e in g_edges:
            covered.add(e["Source_Node_ID"])
            covered.add(e["Target_Node_ID"])

        # cell nodes that are isolated: add a cell_pin edge to a random pin
        for cid in cell_ids:
            if cid not in covered:
                pid = rng.choice(pin_ids)
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": cid, "Target_Node_ID": pid,
                    "Type": "cell_pin",
                    "cp_resistance_ohm": round(rng.uniform(1.0, 50.0), 2),
                    "pn_wire_length_um": None,
                })
                covered.add(cid)
                covered.add(pid)

        # pin nodes that are isolated: add a pin_net edge to a random net
        for pid in pin_ids:
            if pid not in covered:
                nid = rng.choice(net_ids)
                wl = round(rng.uniform(5.0, 200.0), 2)
                total_wl += wl
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": pid, "Target_Node_ID": nid,
                    "Type": "pin_net",
                    "cp_resistance_ohm": None,
                    "pn_wire_length_um": wl,
                })
                covered.add(pid)
                covered.add(nid)

        # net nodes that are isolated: add a pin_net edge from a random pin
        for nid in net_ids:
            if nid not in covered:
                pid = rng.choice(pin_ids)
                wl = round(rng.uniform(5.0, 200.0), 2)
                total_wl += wl
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": pid, "Target_Node_ID": nid,
                    "Type": "pin_net",
                    "cp_resistance_ohm": None,
                    "pn_wire_length_um": wl,
                })
                covered.add(nid)

        # Connectivity pass: merge disconnected components using valid edge types.
        # We use pin nodes as bridges: cell→pin (cell_pin) or pin→net (pin_net).
        all_node_ids = cell_ids + pin_ids + net_ids
        # Build a lookup: node_id → type string
        node_type_map: dict[int, str] = {}
        for cid in cell_ids:
            node_type_map[cid] = "cell"
        for pid in pin_ids:
            node_type_map[pid] = "pin"
        for nid_inner in net_ids:
            node_type_map[nid_inner] = "net"

        uf = _UnionFind(all_node_ids)
        for e in g_edges:
            uf.union(e["Source_Node_ID"], e["Target_Node_ID"])

        comps = uf.components()
        while len(comps) > 1:
            roots = list(comps.keys())
            largest = max(roots, key=lambda r: len(comps[r]))
            # Find a pin in the largest component to use as bridge anchor.
            anchor_pin = next(
                (n for n in comps[largest] if node_type_map[n] == "pin"),
                rng.choice(comps[largest]),
            )
            # Pick any node from a smaller component.
            other_root = next(r for r in roots if r != largest)
            bridge_node = rng.choice(comps[other_root])
            bridge_type = node_type_map[bridge_node]

            if bridge_type == "cell":
                # cell→anchor_pin via cell_pin edge
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": bridge_node, "Target_Node_ID": anchor_pin,
                    "Type": "cell_pin",
                    "cp_resistance_ohm": round(rng.uniform(1.0, 50.0), 2),
                    "pn_wire_length_um": None,
                })
            elif bridge_type == "net":
                # anchor_pin→net via pin_net edge
                wl = round(rng.uniform(5.0, 200.0), 2)
                total_wl += wl
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": anchor_pin, "Target_Node_ID": bridge_node,
                    "Type": "pin_net",
                    "cp_resistance_ohm": None,
                    "pn_wire_length_um": wl,
                })
            else:
                # bridge_node is a pin in the other component.
                # Connect it to anchor_pin via a cell_pin-style bridge:
                # add a cell_pin edge from a cell in the largest component to
                # bridge_node is not valid (cell→pin only).
                # Instead, connect anchor_pin (pin, largest) → bridge_node (pin,
                # other) is not a valid edge type either.
                # Safest: bridge via a net in the largest component:
                #   bridge_node (pin, other) → largest_net (net, largest) [pin_net]
                largest_net = next(
                    (n for n in comps[largest] if node_type_map[n] == "net"),
                    None,
                )
                if largest_net is not None:
                    wl = round(rng.uniform(5.0, 200.0), 2)
                    total_wl += wl
                    g_edges.append({
                        "Graph_ID": gid,
                        "Source_Node_ID": bridge_node, "Target_Node_ID": largest_net,
                        "Type": "pin_net",
                        "cp_resistance_ohm": None,
                        "pn_wire_length_um": wl,
                    })
                    uf.union(bridge_node, largest_net)
                else:
                    # No net in largest component — use anchor_pin→bridge_node via
                    # any net in either component.
                    any_net = next(
                        (n for comp_nodes in comps.values() for n in comp_nodes
                         if node_type_map[n] == "net"),
                        net_ids[0],
                    )
                    wl = round(rng.uniform(5.0, 200.0), 2)
                    total_wl += wl
                    g_edges.append({
                        "Graph_ID": gid,
                        "Source_Node_ID": bridge_node, "Target_Node_ID": any_net,
                        "Type": "pin_net",
                        "cp_resistance_ohm": None,
                        "pn_wire_length_um": wl,
                    })
                    uf.union(bridge_node, any_net)
                comps = uf.components()
                continue

            uf.union(anchor_pin, bridge_node)
            comps = uf.components()

        edge_rows.extend(g_edges)
        graph_rows.append({
            "Graph_ID": gid,
            "num_cells": n_cells,
            "total_wirelength": round(total_wl + rng.uniform(-50, 50), 2),
        })

    return {
        "Parameter": parameter,
        "Node": pd.DataFrame(node_rows),
        "Edge": pd.DataFrame(edge_rows),
        "Graph": pd.DataFrame(graph_rows),
    }


# ── Heterogeneous demo v3 — string Graph_IDs + multi feature groups ──────
# Graph_ID is a string ("G001" … "G030") to exercise non-integer Graph_IDs
# end-to-end. Two node types (CAP, RES). For CAP, three different feature
# groups assigned across graphs:
#     Group A (10 graphs): present X_1, X_2 (NaN for X_3)
#     Group B (10 graphs): present X_1, X_3 (NaN for X_2)
#     Group C (10 graphs): present X_2, X_3 (NaN for X_1)
# RES type uses a fixed feature set (R_1, R_2) for all graphs.
#
# Y target (graph_regression, target_y) is a deterministic linear function
# of the present features so the model can actually learn it:
#     target_y = 2.0 * mean(X_1_after_fillna_0)
#              + 1.5 * mean(X_2_after_fillna_0)
#              + 1.0 * mean(X_3_after_fillna_0)
#              + 0.8 * mean(R_1)
#              + 0.4 * mean(R_2)
#              + small Gaussian noise
# Because NaN columns are fillna(0.0) before scaling, absent features
# contribute 0 to the sum — so the same linear weights work across all
# three CAP feature groups. Tight ranges + small noise keep training loss
# stable and final metrics in a sane band (R^2 well above 0).

def make_hetero_str_gid(n_graphs: int = 30) -> dict[str, pd.DataFrame]:
    rng = random.Random(SEED)

    parameter = pd.DataFrame([
        # CAP node features (three columns; only two are present per graph).
        {"XY": "X", "Level": "Node", "Type": "CAP", "Parameter": "X_1", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "CAP", "Parameter": "X_2", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "CAP", "Parameter": "X_3", "Weight": None},
        # RES node features (consistent across graphs).
        {"XY": "X", "Level": "Node", "Type": "RES", "Parameter": "R_1", "Weight": None},
        {"XY": "X", "Level": "Node", "Type": "RES", "Parameter": "R_2", "Weight": None},
        # cap_res edge feature.
        {"XY": "X", "Level": "Edge", "Type": "cap_res", "Parameter": "weight_cr", "Weight": None},
        # Graph-level Y.
        {"XY": "Y", "Level": "Graph", "Type": "default", "Parameter": "target_y", "Weight": 1.0},
    ])

    feature_groups = ["A", "B", "C"]   # cycles A,B,C,A,B,C,...

    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    graph_rows: list[dict] = []

    global_node_id = 0

    for i in range(n_graphs):
        gid = f"G{i + 1:03d}"
        group = feature_groups[i % 3]

        n_caps = rng.randint(6, 12)
        n_res = rng.randint(4, 8)

        cap_ids = list(range(global_node_id, global_node_id + n_caps))
        global_node_id += n_caps
        res_ids = list(range(global_node_id, global_node_id + n_res))
        global_node_id += n_res

        # Generate CAP features in tight ranges (low variance → easier to learn).
        x1_vals: list[float] = []
        x2_vals: list[float] = []
        x3_vals: list[float] = []
        for cid in cap_ids:
            x1 = round(rng.uniform(0.5, 1.5), 4)
            x2 = round(rng.uniform(0.5, 1.5), 4)
            x3 = round(rng.uniform(0.5, 1.5), 4)
            # Drop one column to NaN based on group so the same node type CAP
            # carries different feature subsets across graphs.
            if group == "A":      # present X_1, X_2 — NaN for X_3
                row_x1, row_x2, row_x3 = x1, x2, None
            elif group == "B":    # present X_1, X_3 — NaN for X_2
                row_x1, row_x2, row_x3 = x1, None, x3
            else:                 # present X_2, X_3 — NaN for X_1
                row_x1, row_x2, row_x3 = None, x2, x3
            node_rows.append({
                "Graph_ID": gid, "Node": cid, "Type": "CAP",
                "X_1": row_x1, "X_2": row_x2, "X_3": row_x3,
                # NaN for the RES-only columns
                "R_1": None, "R_2": None,
            })
            x1_vals.append(row_x1 if row_x1 is not None else 0.0)
            x2_vals.append(row_x2 if row_x2 is not None else 0.0)
            x3_vals.append(row_x3 if row_x3 is not None else 0.0)

        r1_vals: list[float] = []
        r2_vals: list[float] = []
        for rid in res_ids:
            r1 = round(rng.uniform(0.5, 1.5), 4)
            r2 = round(rng.uniform(0.5, 1.5), 4)
            r1_vals.append(r1)
            r2_vals.append(r2)
            node_rows.append({
                "Graph_ID": gid, "Node": rid, "Type": "RES",
                "X_1": None, "X_2": None, "X_3": None,
                "R_1": r1, "R_2": r2,
            })

        # cap_res edges: each CAP connects to one or two RES (deterministic).
        g_edges: list[dict] = []
        for cid in cap_ids:
            partners = rng.sample(res_ids, k=min(rng.randint(1, 2), len(res_ids)))
            for rid in partners:
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": cid, "Target_Node_ID": rid,
                    "Type": "cap_res",
                    "weight_cr": round(rng.uniform(0.1, 1.0), 3),
                })

        # Coverage pass: every RES must appear in at least one edge.
        covered = {e["Source_Node_ID"] for e in g_edges} | {e["Target_Node_ID"] for e in g_edges}
        for rid in res_ids:
            if rid not in covered:
                cid = rng.choice(cap_ids)
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": cid, "Target_Node_ID": rid,
                    "Type": "cap_res",
                    "weight_cr": round(rng.uniform(0.1, 1.0), 3),
                })

        # Connectivity pass: bridge disconnected components via cap_res edges.
        all_node_ids = cap_ids + res_ids
        node_type_map = {n: "CAP" for n in cap_ids}
        node_type_map.update({n: "RES" for n in res_ids})
        uf = _UnionFind(all_node_ids)
        for e in g_edges:
            uf.union(e["Source_Node_ID"], e["Target_Node_ID"])
        comps = uf.components()
        while len(comps) > 1:
            roots = list(comps.keys())
            largest = max(roots, key=lambda r: len(comps[r]))
            anchor_cap = next(
                (n for n in comps[largest] if node_type_map[n] == "CAP"),
                None,
            )
            other_root = next(r for r in roots if r != largest)
            bridge_res = next(
                (n for n in comps[other_root] if node_type_map[n] == "RES"),
                None,
            )
            if anchor_cap is not None and bridge_res is not None:
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": anchor_cap, "Target_Node_ID": bridge_res,
                    "Type": "cap_res",
                    "weight_cr": round(rng.uniform(0.1, 1.0), 3),
                })
                uf.union(anchor_cap, bridge_res)
            else:
                # Fallback: connect any pair to make progress (shouldn't happen).
                a = rng.choice(comps[largest])
                b = rng.choice(comps[other_root])
                g_edges.append({
                    "Graph_ID": gid,
                    "Source_Node_ID": a, "Target_Node_ID": b,
                    "Type": "cap_res",
                    "weight_cr": round(rng.uniform(0.1, 1.0), 3),
                })
                uf.union(a, b)
            comps = uf.components()

        edge_rows.extend(g_edges)

        # Deterministic Y: linear in present feature pool means.
        mean_x1 = sum(x1_vals) / len(x1_vals)
        mean_x2 = sum(x2_vals) / len(x2_vals)
        mean_x3 = sum(x3_vals) / len(x3_vals)
        mean_r1 = sum(r1_vals) / len(r1_vals)
        mean_r2 = sum(r2_vals) / len(r2_vals)
        target = (
            2.0 * mean_x1
            + 1.5 * mean_x2
            + 1.0 * mean_x3
            + 0.8 * mean_r1
            + 0.4 * mean_r2
            + rng.gauss(0.0, 0.05)   # tiny Gaussian noise (std=0.05)
        )
        graph_rows.append({"Graph_ID": gid, "target_y": round(target, 4)})

    return {
        "Parameter": parameter,
        "Node": pd.DataFrame(node_rows),
        "Edge": pd.DataFrame(edge_rows),
        "Graph": pd.DataFrame(graph_rows),
    }


def main() -> None:
    homo_v2 = OUT / "demo_multigraph_homo.v2.xlsx"
    homo_large_v2 = OUT / "demo_multigraph_homo_large.v2.xlsx"
    hetero_v2 = OUT / "demo_multigraph_hetero.v2.xlsx"
    hetero_v3 = OUT / "demo_hetero_multifeature.v3.xlsx"

    _write(homo_v2, make_homo(30))
    _write(homo_large_v2, make_homo(100))

    # Hetero file may be locked if open in Excel; fall back to a temp name.
    try:
        _write(hetero_v2, make_hetero(30))
        print(f"Wrote {hetero_v2}")
    except PermissionError:
        fallback = OUT / "demo_multigraph_hetero.v2.new.xlsx"
        _write(fallback, make_hetero(30))
        print(f"Skipped {hetero_v2} (open in Excel?); wrote {fallback} instead")

    # Hetero v3 (string Graph_IDs + multi-feature groups).
    try:
        _write(hetero_v3, make_hetero_str_gid(30))
        print(f"Wrote {hetero_v3}")
    except PermissionError:
        fallback = OUT / "demo_hetero_multifeature.v3.new.xlsx"
        _write(fallback, make_hetero_str_gid(30))
        print(f"Skipped {hetero_v3} (open in Excel?); wrote {fallback} instead")

    # Also refresh the unversioned homo alias (best-effort; may be locked by Excel).
    try:
        _write(OUT / "demo_multigraph_homo.xlsx", make_homo(30))
    except PermissionError:
        print("Skipped demo_multigraph_homo.xlsx (open in Excel?)")

    print(f"Wrote {homo_v2}")
    print(f"Wrote {homo_large_v2}")


if __name__ == "__main__":
    main()
