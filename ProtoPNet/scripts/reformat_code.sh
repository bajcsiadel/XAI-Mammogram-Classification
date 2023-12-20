modified_files=$(git diff --name-only | grep -E '\.py$' | xargs ls -d 2> /dev/null)

python -m isort $modified_files
python -m black $modified_files
python -m flake8 $modified_files