# Disclosure URL audit — train corpus

- generated_at: 2026-05-18T19:32:41+00:00
- train_rows: 279
- disclosure_trailer_posts: 184 (posts containing 'important disclosures', case-insensitive)
- total_unique_urls: 97

## Top 20 most frequent URLs (all train posts)

| count | url |
|---:|---|
| 180 | https://bit.ly/2JzEDWl |
| 5 | https://www.capitalgroup.com/advisor/insights/articles/10-investment-themes-2022.html |
| 3 | https://bit.ly/2JzEDW |
| 3 | https://www.capitalgroup.com/advisor/insights/articles/2023-bond-market-outlook.html |
| 3 | https://www.capitalgroup.com/advisor/insights/articles/can-tips-help-portfolio-high-inflation.html |
| 3 | https://www.capitalgroup.com/advisor/insights/articles/investment-lessons-40-years-china.html |
| 3 | https://www.capitalgroup.com/institutional/insights/articles/why-fed-behind-curve.html |
| 2 | https://bit.ly/2VNRCdw |
| 2 | https://bit.ly/3fhiLvx |
| 2 | https://bit.ly/3l8JTS3 |
| 2 | https://www.capitalgroup.com/advisor/insights/articles/2022-us-outlook.html |
| 2 | https://www.capitalgroup.com/advisor/insights/articles/3-reasons-60-40-portfolios-comeback.html |
| 2 | https://www.capitalgroup.com/advisor/insights/articles/artificial-intelligence-what-means-investors.html |
| 2 | https://www.capitalgroup.com/advisor/insights/articles/braving-bear-markets-5-lessons-seasoned-investors.html |
| 2 | https://www.capitalgroup.com/advisor/insights/articles/end-era-easy-money.html |
| 2 | https://www.capitalgroup.com/advisor/insights/articles/russia-ukraine-conflict-threatens-global-economy.html |
| 2 | https://www.capitalgroup.com/content/capital-group/us/en/advisor/home/insights/articles/reshoring-supply-chains-what-means-investors.html |
| 2 | https://www.capitalgroup.com/institutional/insights/articles/Europe-consumer-spending-returns.html |
| 2 | https://www.capitalgroup.com/institutional/insights/articles/china-economic-recovery-will-take-time-2022.html |
| 2 | https://www.capitalgroup.com/institutional/insights/articles/dividend-rotation-inflation.html |

## Top 20 most frequent URLs (disclosure-trailer posts only)

| count | url |
|---:|---|
| 180 | https://bit.ly/2JzEDWl |
| 3 | https://bit.ly/2JzEDW |
| 2 | https://bit.ly/2VNRCdw |
| 2 | https://bit.ly/3fhiLvx |
| 2 | https://bit.ly/3l8JTS3 |
| 1 | https://bit.ly/2JzEDWl" |
| 1 | https://bit.ly/2PLUvcf |
| 1 | https://bit.ly/2SGMzdG |
| 1 | https://bit.ly/2TE1a9T |
| 1 | https://bit.ly/2TE75w9 |
| 1 | https://bit.ly/2YMbvjx |
| 1 | https://bit.ly/35dXMoW |
| 1 | https://bit.ly/395LTUH |
| 1 | https://bit.ly/3DYIEfk |
| 1 | https://bit.ly/3e1SDVG |
| 1 | https://bit.ly/3m2o3OF |
| 1 | https://bit.ly/3p9WkMA |
| 1 | https://bit.ly/3pHwHDW |
| 1 | https://bit.ly/3tPvYlY |
| 1 | https://bit.ly/3vEZdqU |

## Canonical disclosure URLs

Threshold: a URL must appear in >= 3 disclosure-trailer posts.

Rationale: URLs that recur as the disclosure-trailer link across multiple posts are stable disclosure pointers, not one-off article promotion links.

| count | url |
|---:|---|
| 180 | https://bit.ly/2JzEDWl |
| 3 | https://bit.ly/2JzEDW |

## Observations

- The dominant disclosure URL `https://bit.ly/2JzEDWl` accounts for the vast majority of disclosure-trailer occurrences.
- The corpus also contains a handful of variants that look like truncations or copy-paste artifacts of the same short URL (e.g. trailing character missing). These pass the mechanical threshold but a human may want to dedupe them before wiring `--include-disclosure` into generation.
