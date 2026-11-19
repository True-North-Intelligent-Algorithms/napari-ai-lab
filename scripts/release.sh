#!/bin/bash
# Tag and publish a release.
#
#   scripts/release.sh            # next alpha: v0.1.0a5 -> v0.1.0a6
#   scripts/release.sh v0.1.0     # or name the version
#   scripts/release.sh --upload   # also build and upload by hand, with uv
#
# Pushing the tag is the release: GitHub Actions (test_and_deploy.yml) builds
# and uploads to PyPI. --upload is the fallback for when CI can't; with
# --check-url it is harmless if CI already did it.
set -euo pipefail
cd "$(dirname "$0")/.."

upload=false
version=""
for arg in "$@"; do
    case $arg in
        --upload) upload=true ;;
        *) version=$arg ;;
    esac
done

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Uncommitted changes. Commit first." >&2
    exit 1
fi

if [ -z "$version" ]; then
    last=$(git tag --list 'v*' --sort=-v:refname | head -1)
    if [[ ! $last =~ ^(.*a)([0-9]+)$ ]]; then
        echo "Can't bump $last. Pass the version." >&2
        exit 1
    fi
    version="${BASH_REMATCH[1]}$((BASH_REMATCH[2] + 1))"
fi

echo "Release $version from $(git log --oneline -1)"
read -rp "Push main and tag? [y/N] " ok
[ "$ok" = y ] || exit 1

git push origin main
git tag "$version"
git push origin "$version"

if $upload; then
    rm -rf dist
    uv build --out-dir dist
    uv publish --check-url https://pypi.org/simple/ dist/*
fi

echo "Done. CI publishes $version; watch the Actions tab."
