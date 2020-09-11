##
# Runs the sphinx docs and commit to gh-pages branch
#

import os

# on branch 'public'
os.system("git checkout public")

# generate sphinx docs
res = os.system("cd docs; make clean; make github")  # result in sphinx_docs/
print(res)

# go to gh-pages, copy docs
res = os.system("git checkout gh-pages")
print(res)

res = os.system("cp -rT sphinx_docs .")
print(res)

# commit and push to github
os.system("git add -u")
os.system("git commit -m 'updated docs'")
os.system("git push github gh-pages")






