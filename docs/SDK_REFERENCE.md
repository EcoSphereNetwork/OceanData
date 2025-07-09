# OceanData SDK Reference

This document summarises the most important functions of the OceanData SDKs.

## Python

### `get_user_identity()`
Returns the current user identifier and optional wallet address derived from environment variables.

### `register_model(name, schema, version)`
Registers a new model and returns a `ModelInfo` object.

### `evaluate_model(model_id, dataset_id)`
Triggers evaluation for a model using Compute-to-Data.

### `retrieve_model_outputs(model_id)`
Returns an `EvaluationResult` with the outputs of the evaluation.

## TypeScript

The TypeScript SDK exposes the same API via generically typed functions in `src/core.ts`.

## Examples

```bash
$ oceandata whoami
$ oceandata register MyModel schema.json 1.0
```
