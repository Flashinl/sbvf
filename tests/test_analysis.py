import unittest

from stockbot import analysis


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class RecommendTests(unittest.TestCase):
    def test_buy_signal_with_uptrend_and_positive_news(self):
        price = Obj(
            ticker="TEST",
            price=100.0,
            market_cap=1_000_000_000.0,
            fifty_two_week_high=160.0,
            fifty_two_week_low=60.0,
            earnings_date=None,
            sector="Technology",
            industry="Semiconductors",
            long_name="Test Semi Inc",
        )
        tech = Obj(sma20=110.0, sma50=105.0, sma200=90.0, rsi14=55.0, trend_score=0.8)
        news = [Obj(title="Company secures large contract", url="", published_at=None, source="", sentiment=0.8)]

        rec = analysis.recommend(price, tech, news, risk="medium")
        self.assertIn(rec.label, {"Buy", "Hold", "Sell"})
        self.assertGreaterEqual(rec.confidence, 1.0)
        self.assertLessEqual(rec.confidence, 99.0)
        # Strong trend + catalyst should typically yield Buy
        self.assertEqual(rec.label, "Buy")
        self.assertIsNotNone(rec.predicted_price)
        self.assertGreater(rec.predicted_price, 0)

    def test_hold_or_sell_when_downtrend(self):
        price = Obj(ticker="TEST", price=50.0, market_cap=50_000_000_000.0, fifty_two_week_high=60.0, fifty_two_week_low=40.0, earnings_date=None)
        tech = Obj(sma20=55.0, sma50=58.0, sma200=59.0, rsi14=65.0, trend_score=-0.7)
        news = []

        rec = analysis.recommend(price, tech, news, risk="low")
        self.assertIn(rec.label, {"Hold", "Sell"})


if __name__ == "__main__":
    unittest.main()

