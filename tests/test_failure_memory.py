from heal_capo.components.failure_memory import FailureCase, FailureMemory


def _memory():
    memory = FailureMemory()

    memory.add(
        FailureCase(
            x="x1",
            output="bad",
            candidate_id="p1",
            failure_type="fairness",
            explanation="bias",
        )
    )
    memory.add(
        FailureCase(
            x="x2",
            output="bad",
            candidate_id="p1",
            failure_type="fairness",
            explanation="bias again",
        )
    )
    memory.add(
        FailureCase(
            x="x3",
            output="wrong",
            candidate_id="p2",
            failure_type="incorrect",
            explanation="wrong answer",
        )
    )

    return memory


def test_memory_add_and_len():
    memory = _memory()

    assert len(memory) == 3
    assert not memory.is_empty()


def test_clusters_by_type():
    memory = _memory()
    clusters = memory.clusters()

    assert set(clusters.keys()) == {"fairness", "incorrect"}
    assert len(clusters["fairness"]) == 2
    assert len(clusters["incorrect"]) == 1


def test_clusters_by_candidate():
    memory = _memory()
    clusters = memory.clusters_by_candidate()

    assert set(clusters.keys()) == {"p1", "p2"}
    assert len(clusters["p1"]) == 2
    assert len(clusters["p2"]) == 1


def test_clusters_by_candidate_and_type():
    memory = _memory()
    clusters = memory.clusters_by_candidate_and_type()

    assert len(clusters[("p1", "fairness")]) == 2
    assert len(clusters[("p2", "incorrect")]) == 1


def test_count_helpers():
    memory = _memory()

    assert memory.count_by_type()["fairness"] == 2
    assert memory.count_by_candidate()["p1"] == 2
    assert memory.count_for_candidate("p1") == 2
    assert memory.count_for_type("incorrect") == 1


def test_recent():
    memory = _memory()
    recent = memory.recent(2)

    assert len(recent) == 2
    assert recent[0].x == "x2"
    assert recent[1].x == "x3"


def test_repeated_failures():
    memory = _memory()
    repeated = memory.repeated_failures(min_count=2)

    assert ("p1", "fairness") in repeated
    assert ("p2", "incorrect") not in repeated


def test_fairness_debt():
    memory = _memory()

    assert memory.fairness_debt() == 2
    assert memory.fairness_debt("p1") == 2
    assert memory.fairness_debt("p2") == 0


def test_risk_debt():
    memory = _memory()

    assert memory.risk_debt() == 1
    assert memory.risk_debt("p1") == 0
    assert memory.risk_debt("p2") == 1


def test_summary_and_rows():
    memory = _memory()

    summary = memory.summary()
    rows = memory.to_rows()

    assert summary["num_failures"] == 3
    assert summary["fairness_debt"] == 2
    assert summary["risk_debt"] == 1
    assert len(rows) == 3
    assert rows[0]["candidate_id"] == "p1"


def test_clear():
    memory = _memory()

    memory.clear()

    assert len(memory) == 0
    assert memory.is_empty()