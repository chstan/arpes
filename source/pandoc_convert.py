#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import subprocess
import os.path


ARPES_ROOT = os.environ['ARPES_ROOT']
DOCS_ROOT = os.path.join(ARPES_ROOT, 'source')
BUILD_ROOT = os.path.join(ARPES_ROOT, 'build')
DESTINATION_ROOT = os.path.join(ARPES_ROOT, 'docs')


def convert_file(src, dst):
    """
    pandoc {src} -f rst -t markdown -o {dst}
    :param src:
    :param dst:
    :return:
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    subprocess.call('pandoc {} -f rst -t gfm -o {}'.format(src, dst), shell=True)


for path, _, files in os.walk(BUILD_ROOT):
    rst_files = [f for f in files if os.path.splitext(f)[1] == '.rst']

    for rst_file in rst_files:
        try:
            destination = Path(os.path.join(path, rst_file)).relative_to(Path(BUILD_ROOT) / 'rst')
        except ValueError:
            destination = Path(os.path.join(path, rst_file)).relative_to(BUILD_ROOT)

        destination = Path(DESTINATION_ROOT) / destination
        destination = os.path.splitext(str(destination))[0] + '.md'
        convert_file(os.path.join(path, rst_file), destination)

subprocess.call('pandoc docs/_sidebar_partial.md docs/toc.md -t gfm > docs/_sidebar.md', shell=True)