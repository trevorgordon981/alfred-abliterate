# Source and artifact boundary

Track the algorithms, prompt sets, structural tests, runbooks, and shell
orchestration in Git. Do not track captured activations, refusal directions,
expanded context corpora, logs, generated reports, baked weights, or model
outputs; those are regenerated artifacts and can be stored by hash elsewhere.

The old Studio directory is not a Git checkout, but a Forgejo repository already
exists. Reconcile this source-only tree onto a review branch without rewriting
the existing public history, and run deterministic CI before any runtime
integration. Do not import the 647 MB working directory wholesale.

Production currently uses a custom Python serving engine. The vMLX/vLLM server
adapter files in this tree are historical experiments, not deployment hooks.
Any custom-engine integration must be reviewed and activated after Byron, with
an explicit rollback, rather than inferred from those legacy files.
