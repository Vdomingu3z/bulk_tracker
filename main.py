from datetime import datetime
from bulk_tracker.bulk_tracker import BulkTracker

# constants
START_DATE = datetime(2021, 11, 7) # first time step is the week after start_date
# END_DATE = datetime(2022, 4, 1)
START_WEIGHT = 158.7
WEEKLY_BULK_RATE_PCT = 0.35 / 100
BULK_RATE = START_WEIGHT * WEEKLY_BULK_RATE_PCT
print(f"bulk rate is: {BULK_RATE} pounds")
NUM_WEEKS = 20
GOAL_WEIGHT = START_WEIGHT + BULK_RATE * NUM_WEEKS
NUMBER_WEEKS_IN_BULK = 5

bulktracker = BulkTracker(
    start_date=START_DATE,
    num_weeks=NUM_WEEKS,
    bulk_rate=BULK_RATE,
    manual_download=False,
)

bulktracker.predict_transform()
bulktracker_df = bulktracker.get_df()
print(bulktracker_df)
bulktracker.plot()
