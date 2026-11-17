# Preserved implementation source

This internal directory contains the implementation files moved from the old
repository layout. Their original hashes are in `docs/backend_source_hashes.json`
at the project root. Public execution is through `run.py` at the project root.

`experiment/backends.py` controls lazy imports of the original model definitions
and tested preprocessing helpers. The public trainers, evaluation and plotting
adapters live outside this directory. Old CLI files here are retained as
implementation references, not alternative public entry points.
