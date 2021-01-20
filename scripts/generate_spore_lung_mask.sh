#!/bin/bash

PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python
SRC_ROOT=/nfs/masi/xuk9/src/lungmask
SRC_NONRIGID_ROOT=/nfs/masi/xuk9/src/Thorax_non_rigid_combine

#set -o xtrace
#${PYTHON_ENV} ${SRC_ROOT}/__main__.py
#set +o xtrace

set -o xtrace
${PYTHON_ENV} ${SRC_NONRIGID_ROOT}/tools/paral_clip_overlay_mask.py
set +o xtrace


