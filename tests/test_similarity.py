import pytest
from src.correlation.similarity.preprocessing import TextPreprocessor
from src.correlation.similarity.methods import levenshtein_similarity, jaccard_similarity, entity_similarity
from src.correlation.similarity.detector import StringSimilarityDetector
from src.models.market import Market

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_preprocessing(preprocessor):
    text = "Will BTC go above $100k?"
    norm = preprocessor.normalize(text)
    assert "bitcoin" in norm # BTC -> bitcoin
    assert "100k" in norm
    # Regex currently keeps $ for price logic
    assert "$" in norm 

def test_levenshtein():
    s1 = "Will Trump win?"
    s2 = "Will Donald Trump win?"
    score = levenshtein_similarity(s1, s2)
    assert score > 0.6 

def test_entity_similarity():
    e1 = {"people": ["Trump"], "custom": {"years": ["2024"]}}
    e2 = {"people": ["Trump", "Biden"], "custom": {"years": ["2024"]}}
    score = entity_similarity(e1, e2)
    # Both matches (people intersection present, year intersection present)
    assert score > 0.5
    
    e3 = {"people": ["Biden"], "custom": {"years": ["2028"]}}
    score_bad = entity_similarity(e1, e3)
    assert score_bad < 0.5

def test_detector_integration():
    m1 = Market(condition_id="1", slug="a", question="Will Trump win 2024?", category="Politics", subcategory="US Presidential", entities={"people": ["Trump"], "custom": {"years": ["2024"]}})
    m2 = Market(condition_id="2", slug="b", question="Will Donald Trump be president in 2024?", category="Politics", subcategory="US Presidential", entities={"people": ["Trump"], "custom": {"years": ["2024"]}})
    m3 = Market(condition_id="3", slug="c", question="Will it rain?", category="Weather", subcategory="Rain", entities={})
    
    detector = StringSimilarityDetector(threshold=0.5)
    results = detector.find_similar_markets([m1, m2, m3])
    
    assert len(results) >= 1
    best = results[0]
    assert best.market_a_id in ["1", "2"]
    assert best.market_b_id in ["1", "2"]
    assert best.overall_score > 0.6
