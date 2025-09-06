from fastapi.testclient import TestClient
from src.fastapi_app import app
import pandas as pd

client = TestClient(app)

def test_predict_endpoint(tmp_path):
    # Create a dummy CSV
    df = pd.DataFrame({
        'title': ["#AI test video"],
        'publishedAt': ["2023-01-01"],
        'viewCount': [10],
        'likeCount': [1],
        'favouriteCount': [0],
        'commentCount': [0]
    })
    file = tmp_path / "test.csv"
    df.to_csv(file, index=False)
    
    # Upload CSV to API
    with open(file, "rb") as f:
        response = client.post("/predict/", files={"file": ("test.csv", f, "text/csv")})
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "trend_reports" in data
    assert isinstance(data["predictions"], list)
    assert isinstance(data["trend_reports"], list)