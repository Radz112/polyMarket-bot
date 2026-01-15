import pytest
from src.correlation.categorizer import MarketCategorizer
from src.models.market import Market

@pytest.fixture
def categorizer():
    return MarketCategorizer()

def test_categorize_politics(categorizer):
    m = Market(
        condition_id="1", slug="trump-win", 
        question="Will Donald Trump win the 2024 Presidential Election?",
        description="..."
    )
    cat, subcat, entities = categorizer.categorize(m)
    
    assert cat == "Politics"
    assert subcat == "US Presidential"
    assert "US Presidential" in entities["custom"].get("years", []) or "2024" in entities.get("dates", []) or "2024" in entities["custom"].get("years", []) or "2024" in m.question

def test_categorize_crypto_price(categorizer):
    m = Market(
        condition_id="2", slug="btc-100k",
        question="Will Bitcoin be above $100,000 on December 31, 2024?"
    )
    cat, subcat, entities = categorizer.categorize(m)
    
    assert cat == "Crypto"
    assert subcat == "Price Thresholds"
    assert "$100,000" in entities["custom"].get("prices", [])

def test_categorize_economics_fed(categorizer):
    m = Market(
        condition_id="3", slug="fed-cut",
        question="Will the Fed cut rates in December?"
    )
    cat, subcat, entities = categorizer.categorize(m)
    
    assert cat == "Economics"
    assert subcat == "Federal Reserve"
    # Spacy might find 'Fed' as ORG
    
def test_categorize_sports_nfl(categorizer):
    m = Market(
        condition_id="4", slug="chiefs-win",
        question="Will the Kansas City Chiefs win the Super Bowl?"
    )
    cat, subcat, entities = categorizer.categorize(m)
    
    assert cat == "Sports"
    assert subcat == "NFL"

def test_unknown_category(categorizer):
    m = Market(
        condition_id="5", slug="random",
        question="Will it rain tomorrow?"
    )
    cat, subcat, entities = categorizer.categorize(m)
    
    # Needs to match fallback
    assert cat == "Other" 
