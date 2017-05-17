# The MIT License (MIT)
# Copyright (c) 2017 Massachusetts Institute of Technology
#
# Author: Cody Rude and Michael Gowanlock
# This software has been created in projects supported by the US National
# Science Foundation and NASA (PI: Pankratius, NSF ACI-1442997, NASA AIST-NNX15AG84G)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from setuptools import setup, Extension
import os

name = 'vdbscan'
version = '0.9.2'
description = 'Variant DBSCAN'
packages = ['vdbscan']
author = 'MITHAGI'
install_requires = ['numpy',
                    'setuptools',]

classifiers=[
    'Topic :: Scientific/Engineering',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',          
    'Programming Language :: C++',
    'Operating System :: POSIX :: Linux']


NTHREADS = os.environ.get('NTHREADS', 4)

print("Setting number of threads to", NTHREADS)


src_dir = 'src'

code_files = ['DBScan.cpp',
              'main.cpp',
              'schedule.cpp',
              'tree_functions.cpp']


module1 = Extension('vdbscan.lib.libSharedVDBSCAN',
                    sources = [os.path.join(src_dir,code) for code in code_files],
                    define_macros = [('NSEARCHTHREADS', NTHREADS)],
                    extra_compile_args = ['-O3', '-fopenmp', ],
                    extra_link_args = ['-fopenmp'],
                    include_dirs=[src_dir],
                    language = 'c++')

setup(name = name,
      version = version,
      description = description,
      packages = packages,
      author = author,
      install_requires = install_requires,
      classifiers=classifiers,

      ext_modules = [module1],          
      package_data={'vdbscan': ['examples/test_dataset.csv','examples/variant_example.py', 'examples/vtest.py']},
      zip_safe = False,
)

