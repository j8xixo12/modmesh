# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon


def _plot3d_text(blocks):
    """Produce Plot3d text for ``blocks`` of ``((nx, ny, nz), origin)``.

    Nodes enumerate x fastest, matching the reader's node-id layout, and
    the node at grid index (i, j, k) sits at physical origin + (i, j, k),
    so a node id can be checked against the coordinates it must carry.
    """
    lines = [str(len(blocks))]
    for (nx, ny, nz), _origin in blocks:
        lines.append('%d %d %d' % (nx, ny, nz))
    for (nx, ny, nz), origin in blocks:
        for axis in range(3):
            values = [(i, j, k)[axis] + origin[axis]
                      for k in range(nz)
                      for j in range(ny)
                      for i in range(nx)]
            lines.append(' '.join('%g' % v for v in values))
    return '\n'.join(lines) + '\n'


def _unit_cube_corners(blocks):
    """The corner-coordinate set of every cell, in the reader's order."""
    cells = []
    for (nx, ny, nz), origin in blocks:
        for k in range(1, nz):
            for j in range(1, ny):
                for i in range(1, nx):
                    cells.append(set(
                        (origin[0] + i - 1 + di,
                         origin[1] + j - 1 + dj,
                         origin[2] + k - 1 + dk)
                        for di in (0, 1)
                        for dj in (0, 1)
                        for dk in (0, 1)))
    return cells


class Plot3dTC(unittest.TestCase):

    def test_plot3d_parsing(self):

        data = """1
2 2 2
0 0 0 0 1 1 1 1
0 0 1 1 0 0 1 1
0 1 0 1 0 1 0 1
"""
        plot3d_instance = solvcon.core.Plot3d(data.encode('utf-8'))
        blk = plot3d_instance.to_block()

        # Check nodes information
        self.assertEqual(blk.nnode, 8)
        # Due to ghost cell and ghost node had been created, the real body
        # had been shifted and start with index 24
        np.testing.assert_almost_equal(blk.ndcrd.ndarray[24:, :].tolist(),
                                       [[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 1.0, 1.0],
                                        [1.0, 0.0, 0.0],
                                        [1.0, 0.0, 1.0],
                                        [1.0, 1.0, 0.0],
                                        [1.0, 1.0, 1.0],
                                        ])
        # Check cells information
        self.assertEqual(blk.ncell, 1)
        self.assertEqual(blk.cltpn.ndarray[6:].tolist(), [5])
        self.assertEqual(blk.clnds.ndarray[6:, :].tolist(),
                         [[8, 0, 2, 6, 4, 1, 3, 7, 5]])

    def test_multi_block_cells_stay_in_their_block(self):
        # Three disjoint blocks; the first has a different shape so a
        # per-block size cannot masquerade as a cumulative offset.
        blocks = [((3, 2, 2), (0.0, 0.0, 0.0)),
                  ((2, 2, 2), (10.0, 0.0, 0.0)),
                  ((2, 2, 2), (20.0, 0.0, 0.0))]
        data = _plot3d_text(blocks)
        blk = solvcon.core.Plot3d(data.encode('utf-8')).to_block()

        self.assertEqual(28, blk.nnode)
        self.assertEqual(4, blk.ncell)
        ndcrd = blk.ndcrd.ndarray[-blk.nnode:, :]
        clnds = blk.clnds.ndarray[-blk.ncell:, :]

        for row, corners in zip(clnds, _unit_cube_corners(blocks)):
            self.assertEqual(8, row[0])
            ids = row[1:9].tolist()
            self.assertEqual(8, len(set(ids)))
            self.assertEqual(corners,
                             set(tuple(crd) for crd in ndcrd[ids]))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
