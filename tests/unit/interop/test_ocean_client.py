from src.interop.ocean_client import OceanClient


def test_publish_dataset_mock() -> None:
    client = OceanClient(mock=True)
    result = client.publish_dataset("demo", {"title": "Demo"}, 1.0)
    assert result["success"]
    assert result["name"] == "demo"
    assert result["price"] == 1.0


def test_did_format() -> None:
    client = OceanClient(mock=True)
    result = client.publish_dataset("demo", {}, 1)
    assert result["did"].startswith("did:op:")
