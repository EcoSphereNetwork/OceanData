from src.interop.datalatte_adapter import DatalatteAdapter


def test_tokenize_mock() -> None:
    adapter = DatalatteAdapter(mock=True)
    result = adapter.tokenize("user", [1, 2, 3])
    assert result["success"]
    assert result["records"] == 3
    assert result["user"] == "user"

def test_token_id_prefix() -> None:
    adapter = DatalatteAdapter(mock=True)
    result = adapter.tokenize("u", [])
    assert result["token_id"].startswith("latte-")
