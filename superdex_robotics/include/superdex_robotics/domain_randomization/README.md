<!--
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DR episode transport

[`DREpisodeList`](domain_randomization_episode.h) transports resolved episode
values between producers and consumers. Experiment configuration, episode
selection and parameter application belong to clients of this library.

## Producer payloads

`provenance` and `metadata` carry producer-owned bookkeeping alongside the
resolved values. They are owned JSON text stored in `DynamicString` fields,
so no borrowed Python object or pointer lifetime is involved.

| Native field | Intended use | JSON contract |
| --- | --- | --- |
| `provenance` | Traceability and lineage, such as a run identifier, configuration hash, worker/rank identity or checkpoint reference. | A JSON object whose values are strings, emitted under the manifest's `provenance` key. An empty native string means unspecified. |
| `metadata` | Carry additional producer fields, such as `profile` or `experiment_overrides`, without adding each one to the native schema. | JSON object text emitted under the `metadata` key in schema 2. An empty native string means no additional fields. |

**Robotics does not interpret or act on these payloads.** They do not select DR
distributions, derive seeds, apply parameters, or prove that a backend applied a
value. The producer defines their meaning; a consumer that understands that
meaning may read them for analysis, replay setup or lineage tracking.

The transport layer parses JSON for validation and serialization only. It checks
the object shapes and string-valued provenance. It does not validate a
run identifier, configuration hash, profile definition or other producer-specific
meaning.

For example, these producer fields can accompany a valid manifest:

```json
{
  "schema_version": 2,
  "spec_version": "example-v1",
  "master_seed": "1",
  "episodes": [],
  "provenance": {
    "run_id": "example-run",
    "worker_rank": "0"
  },
  "metadata": {
    "profile": {
      "name": "example-profile"
    },
    "experiment_overrides": {
      "client_option": true
    }
  }
}
```

On load, `provenance` holds the string-valued object, while `metadata` holds the
`profile` and `experiment_overrides` fields. Their contents are preserved on
save. Producers should encode full-width identifiers as strings and keep resolved
DR values and episode seeds in their modeled fields. Per-episode `applied_keys`
records application coverage separately from producer bookkeeping.

## Round trips

`LoadDREpisodeListFromFile` and `SaveDREpisodeListToFile` use standard Mochi
reflection and support schema 2 only. JSON whitespace and field order may change.
`JsonString` emits both payloads as embedded JSON objects; input may be an object
or JSON object text, following the standard reflection convention. Native
validation checks the parsed payload shape and string-valued provenance.
Unknown top-level and nested fields are rejected; producer extensions belong in
`metadata`. Sparse fields retain their native defaults, which must pass validation.
A missing episode index defaults to -1 and fails validation.

Resolved values use `DynamicArray<real>`: float32 in single-precision native builds
and float64 in double-precision builds. Loading Python-sampled values into a
float32 build can round them; saving those values preserves their native precision
and cannot recover the original binary64 values. JSON values remain ordinary
numbers. Replay consumers should use the same native precision as the recording.

Episode indices are nonnegative int64 JSON integers, through `9223372036854775807`.
They round-trip exactly through native Mochi and Python's standard JSON decoder.
Decoders that store JSON numbers as binary64 cannot preserve every index above
`2^53`; they need an integer-preserving path. Master and episode seeds remain
canonical decimal uint64 strings, separate from episode identity and lineage.

We have prepared an internal tool to convert schema 1 files to schema 2.
Conversion is straightforward but fiddly; we can land the tool if anyone needs it.
