# grdl-sartoolbox

Pure-NumPy reimplementation of the [NGA MATLAB SAR Toolbox](https://github.com/ngageoint/MATLAB_SAR).

## Installation

```bash
pip install grdl-sartoolbox
```

Or install from source:

```bash
git clone https://github.com/geoint-org/grdl-sartoolbox.git
cd grdl-sartoolbox
pip install -e ".[dev]"
```

## Publishing to PyPI

### Dependency Management

All dependencies are defined in `pyproject.toml`. Keep these files synchronized:

- **`pyproject.toml`** — source of truth for versions and dependencies
- **`requirements.txt`** — regenerate with `pip freeze > requirements.txt` after updating `pyproject.toml`
- **`.github/workflows/publish.yml`** — automated PyPI publication (do not edit manually)

### Releasing a New Version

1. Update the `version` field in `pyproject.toml` (semantic versioning: `major.minor.patch`)
2. Update `requirements.txt` if dependencies changed: `pip install -e ".[all,dev]" && pip freeze > requirements.txt`
3. Commit both files
4. Create a git tag: `git tag v0.2.0` (matches version in `pyproject.toml`)
5. Push to GitHub: `git push && git push --tags`
6. Create a GitHub Release from the tag — this triggers the publish workflow automatically

The workflow:
- Builds wheels and source distribution using `python -m build`
- Publishes to PyPI with OIDC authentication (secure, no API keys)
- Artifacts are available at [pypi.org/p/grdl-sartoolbox](https://pypi.org/p/grdl-sartoolbox)

See [CLAUDE.md](CLAUDE.md#dependency-management) for detailed dependency management guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.
