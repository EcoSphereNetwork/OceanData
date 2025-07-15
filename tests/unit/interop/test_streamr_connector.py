from src.interop.streamr_connector import StreamrConnector


def test_create_stream_mock() -> None:
    conn = StreamrConnector(mock=True)
    result = conn.create_stream("demo/stream")
    assert result["success"]
    assert result["stream_id"].startswith("demo/stream")


def test_publish_mock() -> None:
    conn = StreamrConnector(mock=True)
    res = conn.publish("sid", {"value": 1})
    assert res["success"]

def test_publish_size() -> None:
    conn = StreamrConnector(mock=True)
    res = conn.publish("sid", {"hello": "world"})
    assert res["size"] == len(str({"hello": "world"}))
