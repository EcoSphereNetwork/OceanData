from src.interop.dataunion_service import DataUnionService


def test_create_and_distribute() -> None:
    service = DataUnionService(mock=True)
    info = service.create_union("test", ["a", "b"])
    share = service.distribute_revenue(info["union_id"], 10)
    assert share["success"]
    assert share["share"] == 5

def test_distribute_unknown_union() -> None:
    service = DataUnionService(mock=True)
    res = service.distribute_revenue("missing", 10)
    assert not res["success"]
    assert "unknown" in res["error"]
