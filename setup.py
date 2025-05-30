#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="spu_host_cal",  # 包名与目录名一致
    version="1.1.0",
    package_dir={"": "src"},  # 指定包目录为 src
    packages=find_packages(where="src"),  # 从 src 目录中查找包
    install_requires=[],  # 依赖项
    python_requires=">=3.8",  # Python 版本要求
)
