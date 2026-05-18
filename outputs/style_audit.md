# Style audit — train corpus

- generated_at: 2026-05-18T19:04:49+00:00
- train_path: data/inputs/train.xlsx
- train_rows: 279

## overall

- count: 279

| metric | value |
|---|---|
| length min / median / max | 86 / 216 / 412 |
| length p25 / p75 | 179.50 / 260 |
| hashtag presence rate | 52.7% |
| hashtag avg count per post | 0.90 |
| url full capitalgroup.com rate | 30.8% |
| url bit.ly rate | 68.8% |
| url any-url rate | 99.6% |
| url no-url rate | 0.4% |
| disclosure trailer rate | 65.9% |
| emoji rate | 0.0% |
| question mark rate | 31.5% |
| allcaps word rate | 5.4% |

## advisor

- count: 231

| metric | value |
|---|---|
| length min / median / max | 86 / 213 / 409 |
| length p25 / p75 | 177.50 / 254.50 |
| hashtag presence rate | 57.6% |
| hashtag avg count per post | 0.99 |
| url full capitalgroup.com rate | 25.5% |
| url bit.ly rate | 74.5% |
| url any-url rate | 100.0% |
| url no-url rate | 0.0% |
| disclosure trailer rate | 72.3% |
| emoji rate | 0.0% |
| question mark rate | 31.6% |
| allcaps word rate | 5.2% |

## institutional

- count: 34

| metric | value |
|---|---|
| length min / median / max | 139 / 259.50 / 412 |
| length p25 / p75 | 196.75 / 298.50 |
| hashtag presence rate | 23.5% |
| hashtag avg count per post | 0.32 |
| url full capitalgroup.com rate | 61.8% |
| url bit.ly rate | 35.3% |
| url any-url rate | 97.1% |
| url no-url rate | 2.9% |
| disclosure trailer rate | 29.4% |
| emoji rate | 0.0% |
| question mark rate | 20.6% |
| allcaps word rate | 5.9% |

## content

- count: 14

| metric | value |
|---|---|
| length min / median / max | 135 / 224 / 287 |
| length p25 / p75 | 182.75 / 255 |
| hashtag presence rate | 42.9% |
| hashtag avg count per post | 0.79 |
| url full capitalgroup.com rate | 42.9% |
| url bit.ly rate | 57.1% |
| url any-url rate | 100.0% |
| url no-url rate | 0.0% |
| disclosure trailer rate | 50.0% |
| emoji rate | 0.0% |
| question mark rate | 57.1% |
| allcaps word rate | 7.1% |

## Notable differences across tracks

Thresholds: rate metrics with >15 percentage-point spread, length metrics with >25% spread over the lower value.

- hashtag presence rate: advisor 57.6% vs institutional 23.5% (spread 34.0%).
- hashtag avg count per post: advisor 0.99 vs institutional 0.32 (spread 0.66).
- full capitalgroup.com URL rate: institutional 61.8% vs advisor 25.5% (spread 36.2%).
- bit.ly URL rate: advisor 74.5% vs institutional 35.3% (spread 39.2%).
- disclosure trailer rate: advisor 72.3% vs institutional 29.4% (spread 42.9%).
- question mark rate: content 57.1% vs institutional 20.6% (spread 36.6%).
