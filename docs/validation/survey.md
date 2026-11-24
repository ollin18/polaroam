# Against the travel survey

polaroam's outputs for Mexico City are compared with the official
[Origin-Destination Survey 2017](https://www.inegi.org.mx/programas/eod/2017/) (EOD, INEGI) on two
LBS data sets:

- **Dec 2022**: one month, 80 districts covered by the data;
- **Oct 2023–Mar 2024**: six months shared by ITESM, all 85 districts of Mexico City proper.

The suite repeats the checks first made on LBS stop data (expansion factors, OD heat maps,
departure times by trip type, commute distance and time). Every trip-level check is made twice:

- **stays**: a trip is two consecutive stays at least 200 m apart; it leaves when the first stay
  ends;
- **trajectories**: the pings between those two stays; the trip leaves at the first moving ping.

Scope of the comparison:

- Weekdays only (regular weekdays: holidays and the Christmas season are left out), both ends in
  Mexico City districts.
- Only devices with a home location, the population the survey describes (`--all-devices` adds
  the rest).
- The comparison is indicative: 2017 against 2022–24, reported trips against app locations.

The figures below show Oct 2023–Mar 2024; the tables give both data sets.

## Trip types

Each stay has a `location_type` (H home, W work, O other) from the pipeline's `home_work` table.
A trip is classed by the types at its two ends:

| Type | LBS | EOD 2017 |
|---|---|---|
| **HBW** home ↔ work | H → W or W → H; per device and day, the first H → W and the last W → H | Starts at home with purpose *work*, or ends at home after a trip to work; per person, the first and the last |
| **HBO** home ↔ other | One end at home, the other not work | One end at home |
| **NHB** non-home-based | Neither end at home | Neither end at home |
| **all** | Every trip | Every trip |

## Population and expansion factors

![Residents per district against the survey's population](../assets/figures/cdmx_survey_itesm_population_light.png#only-light)
![Residents per district against the survey's population](../assets/figures/cdmx_survey_itesm_population_dark.png#only-dark)

A device's district is the one that contains its home location. A district's expansion factor is
its population divided by the devices living there.

| | Dec 2022 | Oct 2023–Mar 2024 |
|---|---|---|
| Devices with a home | 304,876 | 877,477 |
| Expansion factor, median (range) | 26 (10–66) | 11 (3–373) |
| Residents vs population, Spearman ρ | 0.17 | 0.10 |

The survey's districts were drawn with similar populations (80% hold 75,000–130,000 people), so
there is little to rank. What matters is the spread: a district's share of devices runs from half
to twice its share of people.

The outlier in the six months is San Lorenzo Acopilco (district 060), a mostly forested district
on the western edge with 26,000 people and 71 devices with a home (factor 373).

### By AGEB

The survey stops at districts, but the census goes down to the AGEB (*área geoestadística
básica*, a few blocks). Against the 2020 Census population of Mexico City's 2,431 urban AGEBs:

![Residents per AGEB against the census population](../assets/figures/cdmx_survey_itesm_population_ageb_light.png#only-light)
![Residents per AGEB against the census population](../assets/figures/cdmx_survey_itesm_population_ageb_dark.png#only-dark)

| | Dec 2022 | Oct 2023–Mar 2024 |
|---|---|---|
| Residents vs population, Spearman ρ | 0.61 | 0.64 |
| AGEBs whose share of devices is within ×2 of their share of people | 70% | 75% |
| Expansion factor, median (10th–90th percentile) | 37 (15–77) | 12 (5–23) |
| AGEBs without any device | 90 | 13 |

- **Coverage.** In December some AGEBs lie outside the data's coverage box. With six months,
  only 13 AGEBs have no device, and 99.7% of the homes in the city fall in an urban AGEB.
- **Agreement.** At this scale population varies (from a few hundred to 21,000 people), and devices
  follow it: ρ = 0.64, and three AGEBs in four are within a factor of two.
- **More devices than people.** 27 AGEBs (12,954 devices, 1.5% of the homes) have an expansion
  factor below 1: places where few people live but many devices have their longest night stays.
  The largest is the next point.
- **A likely sink.** One AGEB in the historic centre (19.433, −99.132) holds 8,008 homes for
  947 people. It looks like a default location (one that apps report when they have no fix)
  that the 17 detected sinks did not cover. It is worth checking before using homes at this
  scale.

## Home-work flows

![Home-work OD heat maps and their correlation](../assets/figures/cdmx_survey_itesm_hw_od_light.png#only-light)
![Home-work OD heat maps and their correlation](../assets/figures/cdmx_survey_itesm_hw_od_dark.png#only-dark)

The LBS side counts devices with a home and a work location (`locations`), each expanded by its
home district's factor. The survey side counts workers: each one's household district → the
destination of their first weekday work trip.

| | Dec 2022 | Oct 2023–Mar 2024 |
|---|---|---|
| Workers (LBS devices) | 54,553 | 143,839 |
| log r, all pairs present in both (within / between districts) | 0.73 (0.74 / 0.68) | 0.75 (0.77 / 0.71) |
| Spearman ρ, all pairs | 0.71 | 0.72 |
| Commuting within the home district: EOD / LBS | 19% / 28% | 19% / 27% |

The work locations found in LBS are closer to home than the survey's work places: more commutes
stay within the district, and 22–23% of workers have home and work less than 1 km apart (see
[The commute](#the-commute)). Some are real (people who work near home), some are a regular
place near home (a shop, a relative's house) taken as work.

## Trips between districts

![Trip OD heat maps: survey, stays and trajectories](../assets/figures/cdmx_survey_itesm_trip_od_light.png#only-light)
![Trip OD heat maps: survey, stays and trajectories](../assets/figures/cdmx_survey_itesm_trip_od_dark.png#only-dark)

| Weekday trips, district to district | Dec 2022 stays | Dec 2022 trajectories | Oct–Mar stays | Oct–Mar trajectories |
|---|---|---|---|---|
| Spearman ρ, all pairs | **0.85** | 0.81 | **0.85** | 0.84 |
| log r (within / between districts) | 0.84 (0.45 / 0.82) | 0.81 (0.28 / 0.79) | 0.85 (0.40 / 0.83) | 0.83 (0.31 / 0.82) |
| Trips within one district (EOD 38%) | 29% | 32% | 30% | 30% |

- Both sources rank the flows between districts the same way.
- LBS sees fewer short trips within a district. Trips need two stays 200 m apart, so the short
  walks of the survey (to school, to the market) are partly lost.
- Stays and trajectories agree with each other. Trajectories cover fewer trips: about a quarter
  of the stay pairs give a clean trajectory (pings in between, no GPS jumps).

## When trips start

![Departure hour and trip duration by trip type](../assets/figures/cdmx_survey_itesm_hours_light.png#only-light)
![Departure hour and trip duration by trip type](../assets/figures/cdmx_survey_itesm_hours_dark.png#only-dark)

With stays the departure time is not observed: it lies somewhere between the end of one stay and
the start of the next. Three choices:

- **raw**: the end of the first stay;
- **uniform**: anywhere in the gap;
- **informed**: an hour drawn from the survey's own profile within the gap. It pulls the LBS
  profile towards the survey, so it is **not an independent test**; it is shown because the
  earlier analysis used it.

Trajectories need none of this: the trip leaves at its first moving ping.

Correlation of the hourly profile with the survey's (r; lower DTW = closer):

| Type | Dec 2022 stays raw / uniform / informed | Dec 2022 trajectories | Oct–Mar stays raw / uniform / informed | Oct–Mar trajectories |
|---|---|---|---|---|
| HBW | 0.71 / 0.77 / 0.91 | **0.85** (DTW 0.47) | 0.77 / 0.78 / 0.91 | **0.88** (DTW 0.51) |
| HBO | 0.47 / 0.44 / 0.66 | 0.42 | 0.64 / 0.52 / 0.74 | 0.47 |
| NHB | 0.70 / 0.62 / 0.71 | 0.65 | 0.78 / 0.67 / 0.81 | 0.70 |
| all | 0.59 / 0.58 / 0.73 | 0.58 | 0.72 / 0.64 / 0.80 | 0.64 |

- **Commutes are timed right.** Trajectories reproduce the survey's two HBW peaks (7–8 h and
  18 h) better than any choice based on stays, without borrowing from the survey.
- **The survey's 7:00 peak of other trips is missing.** In the EOD 2017 the HBO peaks at 7:00
  and midday are largely trips to and from school; children and teenagers are likely
  under-represented among devices with location apps (not tested here). The LBS profile of HBO trips rises slowly through
  the morning and peaks in the evening.
- Stays *raw* put more departures at night than the survey has, and fewer in the evening: a stay
  "ends" at its last ping, which can come well before the device leaves.

## How long trips take

| Median minutes | EOD 2017 (reported) | Dec 2022 stays gap | Dec 2022 trajectories | Oct–Mar stays gap | Oct–Mar trajectories |
|---|---|---|---|---|---|
| HBW | 50 | 85 | 39 | 105 | 45 |
| HBO | 20 | 69 | 34 | 88 | 40 |
| NHB | 30 | 52 | 32 | 77 | 40 |
| all | 30 | 67 | 34 | 87 | 41 |

- **The gap between two stays is not a trip duration.** It also holds the time the device was
  not seen: the stay's end and the next stay's start are its last and first pings.
- **Trajectories are close** for commutes (39–45 against 50 minutes). They miss the walk to and
  from the vehicle, and the wait, which the survey's door-to-door times include.
- **HBO trips last longer in LBS** (34–40 against 20 minutes). The survey's short HBO trips
  are the walks LBS misses (see above).

## The commute

![Home-work distance and commute time](../assets/figures/cdmx_survey_itesm_commute_light.png#only-light)
![Home-work distance and commute time](../assets/figures/cdmx_survey_itesm_commute_dark.png#only-dark)

How distance is measured:

- **District to district**: between district centroids, or the district's bounding radius for
  commutes within it, as in the earlier survey analysis. This is the only distance the survey
  allows.
- **LBS home to work location**: the straight line between the device's two locations.

| Median | EOD 2017 | Dec 2022 | Oct 2023–Mar 2024 |
|---|---|---|---|
| Distance to work, district to district | 6.3–6.5 km | 4.7 km | 5.2 km |
| Distance to work, home to work location | | 3.8 km | 3.9 km |
| Commute time, trajectories (EOD reported) | 50 min | 39 min | 45 min |

Commutes in LBS are shorter than in the survey, and more of them stay within the district (see
[Home-work flows](#home-work-flows)).

## How it was computed

The validation code is not part of polaroam: it needs the survey, the census and the district
and AGEB boundaries, none of which the package has to know about. It lives with the survey data,
in a separate validation project. It only reads the tables a pipeline run writes (`home_work`,
`locations`, `trajectories`), so the same checks can be redone for any city with a travel survey.

Mode shares and the hour profile by mode are on the
[Mexico City page](../examples/mexico-city.md#checked-against-the-official-survey).
