# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Launch Model Viewer, optionally opening one GLB."""

import argparse

from ... import apps, pilot  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()
    pilot.require('Model Viewer')
    controller = pilot.controller
    controller.build()
    viewer = controller.apps['modelview']
    if args.path:
        viewer.load_path(args.path)
    else:
        viewer.open()
    controller.launch()


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
