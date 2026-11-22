# Validation

PolaRoam is checked three ways:

<div class="grid cards" markdown>

-   :material-scale-balance:{ .lg .middle } **Parity with gc-lbs**

    ---

    Against the Apache Sedona port of the BigQuery pipeline, on the example data and on Mexico
    City, stage by stage.

    [:octicons-arrow-right-24: Parity](parity.md)

-   :material-timer-outline:{ .lg .middle } **Benchmarks**

    ---

    Time and memory from 11M to 1.66B pings, on CPU and GPU, all at once and day by day.

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md)

-   :material-test-tube:{ .lg .middle } **Test suite**

    ---

    35 tests: independent oracles (SQL port, scikit-learn, SciPy, recursive Douglas-Peucker),
    known-answer trips, and equality across every route.

    [:octicons-arrow-right-24: Tests](testing.md)

</div>
