# Event-table dictionary

Each row is one saved configuration. Every interval event refers to the interval
ending at that row's timestamp. A CSV contains 12,000 rows and 71 columns.

| Field | Definition |
| --- | --- |
| `frame_index` | Zero-based saved-frame index |
| `step` | LAMMPS integration step |
| `time_ns` | `step × 10^-6` ns for the 1 fs integration step |
| `interval_start_ns`, `interval_end_ns` | Endpoints of the observed interval |
| `interval_duration_ns` | 0 for baseline, 0.01 ns for each subsequent interval |
| `interval_observed` | 0 for the baseline, 1 for observed intervals |

The following fields repeat for species prefixes `water`, `Li`, `Cl`, `Mg`.

| Suffix | Definition |
| --- | --- |
| `legacy_forward`, `legacy_reverse` | Positive/negative crossings using the original training-label counting rule |
| `legacy_net` | Forward minus reverse within this interval |
| `legacy_cumulative` | Sum of legacy net events since the first saved frame |
| `plane_forward`, `plane_reverse`, `plane_net`, `plane_cumulative` | Corresponding upper-plane counts with periodic correction and without the legacy common-slab restriction |
| `transit_forward`, `transit_reverse`, `transit_net`, `transit_cumulative` | Corresponding completed transfers between the two reservoir regions |
| `feed` | Current population below z = 39.803101 Å |
| `membrane` | Current population between and including the two boundary planes |
| `permeate` | Current population above z = 53.043098 Å |
| `wrap_net` | Signed periodic z-boundary crossings; positive for wrapping from high z to low z |

## Legacy rule

Convert each z coordinate to float32, as in the original Python calculation.
The particle must be present in the open interval
`39.803101 < z < 58.043098 Å` in **both** adjacent saved frames. Its change in
the Boolean state `z > 53.043098 Å` is +1, -1 or zero. Accumulate signed events.
These values are the appropriate recount comparator for `FWater`, `FLi`,
`FCl`, `FMg` in the paper. A long saved-frame jump can cross the boundary while
failing the common-slab restriction, so the legacy and plane totals can differ.

## Plane rule and periodic mass balance

Normalize coordinates into the recorded z box. Determine the nearest-image
displacement using half the box length. Count crossings of the periodic image
of the upper plane along that displacement. Forward and reverse crossings are
stored separately; reverse crossings subtract from the net value. Repeated
recrossings are not deduplicated by particle identity.

For each species and every observed interval:

```text
permeate(now) - permeate(previous)
  = plane_forward - plane_reverse - wrap_net
feed + membrane + permeate = constant species population
```

This equation applies to `plane`, not the restricted legacy counter or completed
transit counter. Mass-balance validation therefore does not prove that the three
definitions have equal totals, or establish stationarity.

## Completed-transfer rule

Remember the most recently occupied reservoir side while the particle is within
the membrane region. Count one completed transit when it reaches the opposite
reservoir. Leaving and returning to the same reservoir is not a completed
transit. A periodic reservoir wrap resets that remembered side; it is not
counted as a membrane transit. Particles first observed inside the membrane
have no known preceding reservoir until they enter one.

## Block tables

The 12 nonoverlapping nominal 10 ns blocks assign an interval to the block
containing its ending step, with an interval ending exactly at a block boundary
assigned to the preceding block. The first block covers observed endpoints
0.01–10 ns (duration 9.99 ns); the others have duration 10 ns. Actual start,
end and duration are exported. Forward, reverse and signed net counts are
summed separately for every species and definition. A rate in counts/ns can
be obtained by dividing a net count by its exported duration. No area or
pressure normalization is performed and no uncertainty model is assumed.

## Screening descriptors

The complete grid is 4 × 7 × 7 = 196 systems. `Hydrophilicity` is the dimensionless framework charge-scaling factor (0.5, 0.8, 1.0, 1.2). `Charge` is the net framework charge in elementary-charge units (-9, -6, -3, 0, 3, 6, 9). `Sigma` is the Lennard-Jones sigma parameter in Å (1.346421 to 4.346421 in 0.5 Å increments), not a direct geometric pore diameter. Source directory sigma suffixes are offsets from 2.846421 Å.

## Summary and version comparison

`recount_summary.csv` contains a row per case. Suffix `_paper` is the unchanged
paper model label; `_legacy`, `_plane`, `_transit` are the actual recounts;
`_difference = _legacy - _paper`. `row_id_0based` maps directly to the original
training CSV. `training_vs_recount_differences.csv` contains only cases where
at least one legacy recount differs from the paper label. A difference must
not be silently corrected by rescaling an event series or replacing a label.

`source_sha256` hashes the complete atomic trajectory, not just its metadata.
The archived trajectory paths are relative to the user-supplied raw root.
