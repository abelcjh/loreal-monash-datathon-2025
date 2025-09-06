from src.trend_report import generate_trend_report

def test_generate_report():
    hashtag = "#AI"
    metrics = {
        "mentions": [100, 120, 150],
        "views": [1000, 1500, 2000],
        "likes": [50, 80, 120],
        "comments": [10, 15, 20]
    }
    
    report = generate_trend_report(hashtag, metrics)
    
    # The report should be a string
    assert isinstance(report, str)
    
    # It should mention the hashtag
    assert hashtag in report
    
    # It should classify the trend status
    assert any(word in report for word in ["Emerging", "Peaking", "Decaying"])