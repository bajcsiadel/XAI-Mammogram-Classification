project_root=`git rev-parse --show-toplevel`
modified_files=`git diff --name-only | grep -E '\.py$' | sed 's,^,'"$project_root"'/,' | xargs ls -d 2> /dev/null`

python -m isort $modified_files
python -m black $modified_files
python -m flake8 $modified_files