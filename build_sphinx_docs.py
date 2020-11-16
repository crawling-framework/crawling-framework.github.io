##
# Runs the sphinx docs and commit to gh-pages branch
#

import os
import sys

# on branch 'public'
res = os.system("git checkout public")
if res != 0:
    sys.exit(res)

# generate sphinx docs
res = os.system("cd docs; make clean; make github")  # result in sphinx_docs/
if res != 0:
    sys.exit(res)

# go to gh-pages, copy docs
res = os.system("git checkout gh-pages")
if res != 0:
    sys.exit(res)

res = os.system("cp -rT sphinx_docs .")
if res != 0:
    sys.exit(res)

# commit and push to github
os.system("git add -u")
os.system("git commit -m 'updated docs'")
os.system("git push github gh-pages")






