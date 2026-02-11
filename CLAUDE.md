# grdl-sartoolbox Development Guide

## Project Overview

Pure-NumPy reimplementations of NGA MATLAB SAR Toolbox

## Architecture

- **Base classes** come from `grdl` (dependency): `ImageTransform`, `BandwiseTransformMixin`
- **Versioning** via `@processor_version('x.y.z')` from `grdl.image_processing.versioning`
- **Tags** via `@processor_tags(modalities=[...], category=...)` from `grdl.image_processing.versioning`
- **Parameters** via `typing.Annotated` with `Range`, `Options`, `Desc` from `grdl.image_processing.params`
- **Categories** from `grdl.vocabulary.ProcessorCategory`

**Every processor in grdl-sartoolbox must have all three metadata annotations** (`@processor_version`, `@processor_tags`, and `Annotated` parameter declarations). This metadata drives grdl-runtime catalog discovery and grdk widget UI generation. See the Processor Pattern below.


## Code Style

- **Line length**: 100 characters (black, ruff)
- **Python**: >= 3.11
- **Type hints** on all public methods
- **NumPy-style docstrings** (Parameters, Returns, Raises, Examples)
- **Imports**: fail-fast at module level (no lazy imports in hot paths)

### File Headers

Every Python file includes:

```python
# -*- coding: utf-8 -*-
"""
Title - one line.

Purpose and description.

Attribution
-----------
Original source/author info.

Dependencies
------------
List of imports with usage notes.

Author
------
Name

License
-------
MIT License
Copyright (c) 2024 geoint.org

Created
-------
YYYY-MM-DD

Modified
--------
YYYY-MM-DD
"""
```

### Test Conventions

- One test class per processor: `TestProcessorName`
- Test algorithmic correctness, not implementation details
- Use synthetic images from fixtures (deterministic, seed=42)
- Verify: output shape, dtype, value ranges, edge cases, parameter validation

## Dependencies

- `grdl` — base classes, versioning, parameter system, vocabulary
- `numpy` — array operations
- `scipy` — ndimage filters (background, binary, edges, filters, find_maxima, threshold)

## Git Practices

- Conventional commit messages
- One logical change per commit
- Never commit `.pyc`, `__pycache__/`, or IDE files
