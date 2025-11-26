from __future__ import annotations
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Text, Enum, ForeignKey, JSON, Date, Index
from sqlalchemy.orm import relationship
from .db import Base
from sqlalchemy import func
import enum


# Enums for type safety
class SubscriptionTier(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ANNUAL = "annual"


class ExplanationMode(str, enum.Enum):
    BEGINNER = "beginner"
    EXPERT = "expert"


class SignalType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class FactorCategory(str, enum.Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    BUSINESS_CATALYST = "business_catalyst"
    SENTIMENT = "sentiment"
    MARKET_CONTEXT = "market_context"


class FactorDirection(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class CatalystEventType(str, enum.Enum):
    EARNINGS = "earnings"
    FDA_DECISION = "fda_decision"
    MERGER_VOTE = "merger_vote"
    REGULATORY_RULING = "regulatory_ruling"
    PRODUCT_LAUNCH = "product_launch"
    CONTRACT_DECISION = "contract_decision"
    COURT_RULING = "court_ruling"
    GUIDANCE_UPDATE = "guidance_update"
    PARTNERSHIP = "partnership"
    ACQUISITION = "acquisition"


class ImpactLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImpactDirection(str, enum.Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    UNCERTAIN = "uncertain"


class TradeAction(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


# ============================================================================
# USER & AUTHENTICATION
# ============================================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    role = Column(String(32), default="user", nullable=False)

    # Subscription info
    subscription_tier = Column(Enum(SubscriptionTier), default=SubscriptionTier.FREE, nullable=False)
    subscription_start = Column(Date, nullable=True)
    subscription_end = Column(Date, nullable=True)
    stripe_customer_id = Column(String(255), nullable=True, index=True)
    stripe_subscription_id = Column(String(255), nullable=True)

    # Preferences
    explanation_mode = Column(Enum(ExplanationMode), default=ExplanationMode.BEGINNER, nullable=False)
    email_reports_enabled = Column(Boolean, default=True, nullable=False)
    push_notifications_enabled = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login_at = Column(DateTime, nullable=True)

    # Relationships
    watchlist_items = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    paper_trades = relationship("PaperTrade", back_populates="user", cascade="all, delete-orphan")
    explanation_feedback = relationship("ExplanationFeedback", back_populates="user", cascade="all, delete-orphan")


# ============================================================================
# PREDICTIONS & EXPLANATIONS
# ============================================================================

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    stock_symbol = Column(String(10), nullable=False, index=True)
    prediction_date = Column(Date, nullable=False, index=True)

    # Prediction details
    signal = Column(Enum(SignalType), nullable=False)
    predicted_change_pct = Column(Float, nullable=True)  # Expected return %
    predicted_price = Column(Float, nullable=True)  # Target price
    current_price = Column(Float, nullable=True)  # Price at prediction time

    # Confidence & Risk
    confidence_score = Column(Float, nullable=False)  # 0.0 to 1.0
    confidence_interval_low = Column(Float, nullable=True)  # Lower bound %
    confidence_interval_high = Column(Float, nullable=True)  # Upper bound %
    risk_score = Column(Integer, nullable=True)  # 1-10 scale

    # Risk-adjusted metrics
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)

    # Model metadata
    model_version = Column(String(50), nullable=True)
    ensemble_agreement = Column(Float, nullable=True)  # % of models agreeing

    # Actual outcomes (filled later)
    actual_price_7d = Column(Float, nullable=True)
    actual_price_30d = Column(Float, nullable=True)
    actual_change_7d = Column(Float, nullable=True)
    actual_change_30d = Column(Float, nullable=True)
    accuracy_score = Column(Float, nullable=True)  # How accurate was prediction

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    explanations = relationship("PredictionExplanation", back_populates="prediction", cascade="all, delete-orphan")
    confidence_breakdown = relationship("ConfidenceBreakdown", back_populates="prediction", uselist=False, cascade="all, delete-orphan")
    historical_patterns = relationship("HistoricalPatternMatch", back_populates="prediction", cascade="all, delete-orphan")
    feedback = relationship("ExplanationFeedback", back_populates="prediction", cascade="all, delete-orphan")

    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_symbol_date', 'stock_symbol', 'prediction_date'),
    )


class PredictionExplanation(Base):
    __tablename__ = "prediction_explanations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)
    factor_rank = Column(Integer, nullable=False)  # 1 = top factor, 2 = second, etc.

    # Factor details
    factor_category = Column(Enum(FactorCategory), nullable=False)
    factor_name = Column(String(255), nullable=False)
    factor_weight = Column(Float, nullable=False)  # % influence (0-100)
    factor_direction = Column(Enum(FactorDirection), nullable=False)

    # Explanations (both modes)
    expert_explanation = Column(Text, nullable=False)
    beginner_explanation = Column(Text, nullable=False)

    # Supporting data with actual numbers
    supporting_data = Column(JSON, nullable=True)  # Store metrics, contract details, etc.
    source_url = Column(Text, nullable=True)  # Link to news/SEC filing
    event_date = Column(Date, nullable=True)  # When business event occurred

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    prediction = relationship("Prediction", back_populates="explanations")


class ConfidenceBreakdown(Base):
    __tablename__ = "confidence_breakdown"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, unique=True, index=True)

    # Breakdown components
    model_agreement_score = Column(Float, nullable=False)  # 0-1, how many models agree
    historical_accuracy = Column(Float, nullable=False)  # Past accuracy on similar patterns
    data_quality_score = Column(Float, nullable=False)  # Completeness of input data
    volatility_adjustment = Column(Float, nullable=False)  # Market volatility impact
    final_confidence = Column(Float, nullable=False)  # Combined confidence

    # Model-specific confidences
    lstm_confidence = Column(Float, nullable=True)
    xgboost_confidence = Column(Float, nullable=True)
    transformer_confidence = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    prediction = relationship("Prediction", back_populates="confidence_breakdown")


class HistoricalPatternMatch(Base):
    __tablename__ = "historical_pattern_matches"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)

    # Pattern match details
    historical_date = Column(Date, nullable=False)  # When similar pattern occurred
    historical_outcome = Column(String(50), nullable=False)  # e.g., "+6.2% in 30 days"
    pattern_similarity_score = Column(Float, nullable=False)  # 0-1, how similar
    outcome_matched = Column(Boolean, nullable=True)  # Did it match prediction?

    # Context
    historical_context = Column(Text, nullable=True)  # What was happening then

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    prediction = relationship("Prediction", back_populates="historical_patterns")


# ============================================================================
# UPCOMING CATALYSTS (Business Events Calendar)
# ============================================================================

class UpcomingCatalyst(Base):
    __tablename__ = "upcoming_catalysts"

    id = Column(Integer, primary_key=True, index=True)
    stock_symbol = Column(String(10), nullable=False, index=True)

    # Event details
    event_type = Column(Enum(CatalystEventType), nullable=False)
    event_name = Column(String(255), nullable=False)
    expected_date = Column(Date, nullable=False, index=True)
    date_confirmed = Column(Boolean, default=False, nullable=False)

    # Impact assessment
    potential_impact = Column(Enum(ImpactLevel), nullable=False)
    impact_direction = Column(Enum(ImpactDirection), nullable=False)

    # Additional context
    details = Column(Text, nullable=True)
    source_url = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Composite index
    __table_args__ = (
        Index('idx_symbol_expected_date', 'stock_symbol', 'expected_date'),
    )


# ============================================================================
# WATCHLIST
# ============================================================================

class Watchlist(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    stock_symbol = Column(String(10), nullable=False)

    # Preferences
    alert_on_prediction = Column(Boolean, default=True, nullable=False)
    alert_threshold = Column(Float, nullable=True)  # Only alert if confidence > X

    # Timestamps
    added_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="watchlist_items")

    # Composite unique constraint and index
    __table_args__ = (
        Index('idx_user_symbol', 'user_id', 'stock_symbol', unique=True),
    )


# ============================================================================
# PAPER TRADING
# ============================================================================

class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    stock_symbol = Column(String(10), nullable=False, index=True)

    # Trade details
    action = Column(Enum(TradeAction), nullable=False)
    quantity = Column(Integer, nullable=False)
    price_per_share = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)

    # Context (why was this trade made?)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)  # If based on AI prediction
    reason = Column(Text, nullable=True)  # Brief explanation

    # Timestamps
    trade_date = Column(DateTime, server_default=func.now(), nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="paper_trades")


class PaperPortfolio(Base):
    __tablename__ = "paper_portfolios"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True, index=True)

    # Portfolio state
    cash_balance = Column(Float, default=100000.0, nullable=False)  # Start with $100K
    total_value = Column(Float, default=100000.0, nullable=False)
    total_return_pct = Column(Float, default=0.0, nullable=False)

    # Performance metrics
    total_trades = Column(Integer, default=0, nullable=False)
    winning_trades = Column(Integer, default=0, nullable=False)
    losing_trades = Column(Integer, default=0, nullable=False)
    win_rate = Column(Float, nullable=True)

    # Risk metrics
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


# ============================================================================
# FEEDBACK & IMPROVEMENT
# ============================================================================

class ExplanationFeedback(Base):
    __tablename__ = "explanation_feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)

    # Feedback
    helpful = Column(Boolean, nullable=False)  # Thumbs up/down
    feedback_text = Column(Text, nullable=True)  # Optional comment

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="explanation_feedback")
    prediction = relationship("Prediction", back_populates="feedback")


# ============================================================================
# STOCK DATA CACHE (avoid repeated API calls)
# ============================================================================

class StockDataCache(Base):
    __tablename__ = "stock_data_cache"

    id = Column(Integer, primary_key=True, index=True)
    stock_symbol = Column(String(10), nullable=False, index=True)
    data_type = Column(String(50), nullable=False)  # 'price', 'fundamentals', 'news', etc.

    # Cached data
    data = Column(JSON, nullable=False)

    # Cache metadata
    expires_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Composite index
    __table_args__ = (
        Index('idx_symbol_type_expires', 'stock_symbol', 'data_type', 'expires_at'),
    )


# ============================================================================
# HISTORICAL ACCURACY TRACKING
# ============================================================================

class AccuracyMetrics(Base):
    __tablename__ = "accuracy_metrics"

    id = Column(Integer, primary_key=True, index=True)
    stock_symbol = Column(String(10), nullable=False, index=True)
    calculation_date = Column(Date, nullable=False, index=True)

    # Rolling accuracy windows
    accuracy_7d = Column(Float, nullable=True)  # 7-day accuracy
    accuracy_30d = Column(Float, nullable=True)  # 30-day accuracy
    accuracy_90d = Column(Float, nullable=True)  # 90-day accuracy

    # Performance metrics
    total_predictions = Column(Integer, nullable=False)
    correct_direction = Column(Integer, nullable=False)
    avg_error_pct = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Composite index
    __table_args__ = (
        Index('idx_symbol_calc_date', 'stock_symbol', 'calculation_date'),
    )
