from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, Project
from feast.value_type import ValueType
from feast.types import Int64, Float64, Float32

# Define the project
project = Project(name="churn_repo", description="Churn prediction feature store")

# Define the entity - this is the "primary key" Feast uses to look up features
customer = Entity(name="customer", join_keys=["customer_id"], value_type=ValueType.STRING)
# Point Feast at our converted parquet file
customer_features_source = FileSource(
    name="customer_features_source",
    path="data/train_features.parquet",
    timestamp_field="event_timestamp",
)

# Define the feature view - the schema of features Feast will serve
customer_features_fv = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="annual_income", dtype=Float64),
        Field(name="dependents", dtype=Int64),
        Field(name="tenure", dtype=Int64),
        Field(name="senior_citizen", dtype=Int64),
        Field(name="monthlycharges", dtype=Float64),
        Field(name="totalcharges", dtype=Float64),
        Field(name="num_services", dtype=Int64),
        Field(name="has_phone_service", dtype=Int64),
        Field(name="has_internet_service", dtype=Int64),
        Field(name="has_online_security", dtype=Int64),
        Field(name="has_online_backup", dtype=Int64),
        Field(name="has_device_protection", dtype=Int64),
        Field(name="has_tech_support", dtype=Int64),
        Field(name="has_streaming_tv", dtype=Int64),
        Field(name="has_streaming_movies", dtype=Int64),
        Field(name="customer_satisfaction", dtype=Float64),
        Field(name="num_complaints", dtype=Float64),
        Field(name="num_service_calls", dtype=Int64),
        Field(name="late_payments", dtype=Int64),
        Field(name="avg_monthly_gb", dtype=Float64),
        Field(name="days_since_last_interaction", dtype=Int64),
        Field(name="credit_score", dtype=Float64),
        Field(name="signup_year", dtype=Int64),
        Field(name="signup_month", dtype=Int64),
    ],
    online=True,
    source=customer_features_source,
    tags={"team": "churn_model"},
)