#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

from setuptools import setup, find_packages

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="tbsim",
    packages=[package for package in find_packages() if package.startswith("tbsim")],
    install_requires=[
        "pytorch-lightning==1.8.3.post0",
        "numpy==1.23.4",  # need to manually update numpy version to (1.21.4) due to conflict with l5kit's requirement
        "wandb",
        "pyemd",
        "h5py",
        "imageio-ffmpeg==0.4.9",
        "casadi",
        "protobuf==3.20.1", # new version might cause error
        "einops==0.6.0",
        "torchtext==0.14.1",  # weird pytorch-lightning dependency bug
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8",
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
