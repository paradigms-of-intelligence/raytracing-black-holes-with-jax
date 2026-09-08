#!/bin/sh

# Copyright 2026 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR=$(readlink -f $(dirname "${0}"))
cd $SCRIPT_DIR

echo "Rebuilding 'extra material' PDFs from Markdown..."
pandoc -s -f markdown -t pdf coordinate_transformations/ct_howto.md \
       -o coordinate_transformations/ct_howto.pdf

echo "Rebuilding main article from TeX..."
# This is slightly tricky - all code files should carry a license
# header, but here, this code serves a dual purpose of also being
# part of the paper. We hence have to transform the source before
# producing PDF from TeX. We do this by copying it to a
# "working directory" that gets deleted at the end and translating
# code there. This step assumes that perl5 is available, but
# since most TeX installations depend on perl, this usually is satisfied.
(cp -a tex tex_work; cd tex_work;
 perl -i.orig -0777 -pe 's/^(#[^\n]*\n)*//g' code/*.py;
 pdflatex main.tex; bibtex main; pdflatex main.tex;
 cp main.pdf ../tex; cd ..; rm -rf tex_work)

