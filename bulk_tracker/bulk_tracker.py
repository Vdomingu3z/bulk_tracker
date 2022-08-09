import datetime
import pdb
from myfitnesspal_client.myfitnesspal_client import MyFitnessPalClient

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dateutil.parser

import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression


class BulkTracker:
    def __init__(
        self,
        start_date,
        num_weeks,
        bulk_rate,
        manual_download,
    ):

        self.start_date = start_date
        self.num_weeks = num_weeks
        self.bulk_rate = bulk_rate
        self.manual_download = manual_download
        self.dateindex = pd.date_range(
            start=self.start_date, periods=self.num_weeks, freq="W"
        )
        self.init_df()

    def calculate_end_weight(self, start_weight, bulk_rate, num_weeks):
        end_weight = start_weight + bulk_rate * num_weeks

        return end_weight

    def init_df(self):
        # build base df
        self.df = pd.DataFrame({"Date": self.dateindex})
        self.df.set_index("Date", inplace=True)

    def calculate_weekly_goals(self):

        self.end_weight = self.calculate_end_weight(
            self.start_weight, self.bulk_rate, self.num_weeks
        )

        # build weekly goal array
        weekly_weight_goal = np.arange(
            start=self.start_weight, stop=self.end_weight, step=self.bulk_rate
        )

        # remove the last value if weekly_weight_goal vector is greater than df
        if len(weekly_weight_goal) > len(list(self.dateindex)):
            weekly_weight_goal = weekly_weight_goal[:-1]

        # build lower and upper error bounds as 0.25% of weekly weight goal.
        lower_bound = weekly_weight_goal - 0.0025 * weekly_weight_goal
        upper_bound = weekly_weight_goal + 0.0025 * weekly_weight_goal

        return weekly_weight_goal, lower_bound, upper_bound

    def get_weekly_data_df(self) -> pd.DataFrame:
        """
        Returns the weekly average of weight and calorie data.

        Weekly averages are calculated as the average of data from Sunday through Saturday.
        The data is indexed with the First date in the sample, which is the Sunday for each week.

        For example: 
        Data for 11/7/2021 is comprised of the average of data from 11/7/2021-11/13/2021
        """
        
        # get start_date one week prior due to resampling
        mfp_start_date = self.start_date
        mfpc = MyFitnessPalClient(mfp_start_date, manual_download=self.manual_download)
        myfitnesspal_df = mfpc.get_myfitnesspal_df()

        # Take weekly averages of each column indepdently.
        # This is done in order to avoid dropping calorie data, on days where
        # there is no weight data.
        weight_data = myfitnesspal_df.weight.dropna()
        weight_data = weight_data.resample("W-SAT").mean()
        nutrition_data = myfitnesspal_df.calories.dropna()
        nutrition_data = nutrition_data.resample("W-SAT").mean()

        # combine columns into single df
        weekly_data_df = pd.concat([weight_data, nutrition_data], axis=1)
        
        # Shift indices to be the Sunday at the start of each sample, 
        # as oppose to the Saturday at the end of the sample.
        weekly_data_df.index = weekly_data_df.index - datetime.timedelta(6)

        return weekly_data_df

    def predict_transform(self) -> None:
        """
        Transforms the dataframe by setting columns of weight and calorie data obtained
        from myfitnesspal, as well as weekly goals.

        Performs a Linear regression on the weight data obtained from myfitnesspal,
        and predicts the weight for the time period using the data.
        """
        # Calculate weeks from start.
        self.df["weeks_from_start"] = (self.df.index - self.df.index[0]).days / 7

        # Get weekly data df, which is calorie and weight data from myfitnesspal.
        weekly_data_df = self.get_weekly_data_df().dropna()
        self.start_weight = weekly_data_df["weight"].values[0]

        # Generate weight goals, bounds, set columns to df.
        weekly_weight_goals, lower_bound, upper_bound = self.calculate_weekly_goals()
        self.df["weekly_weight_goal"] = weekly_weight_goals
        self.df["lower_bound"] = lower_bound
        self.df["upper_bound"] = upper_bound

        # Concatenate all of the data to the base df.
        self.df = pd.concat([weekly_data_df, self.df], axis=1)

        # Calculate weekly differences.
        self.df["weekly weight differences"] = self.calculate_weight_differences()

        # Fit and predict weight using linear regression model.
        self._fit()
        weight_predictions = self._predict()

        # Set predictions column.
        self.df["weight_predictions"] = weight_predictions

    def calculate_weight_differences(self) -> pd.Series:
        """
        Calculate the rolling difference between 2 weeks worth of data.
        """
        # NOTE: this returns a Series of weight differences calculated from the 'Weight' column.
        return self.df.weight.diff().values

    def calculate_weeks_in_bulk(self, weight_col) -> int:
        """Return the number of weeks in the bulk."""
        number_weeks_in_bulk = len(weight_col.values)

        return number_weeks_in_bulk

    def _fit(self) -> None:
        """Fit the linear regression model."""
        base_reg_df = self.df

        # drop NaN values.
        weight_col = base_reg_df["weight"].dropna()
        weeks_col = base_reg_df["weeks_from_start"][0:len(weight_col)]
        
        # Generate X and y vectors.
        X = weeks_col.values.reshape(-1, 1)
        y = weight_col.values.reshape(-1, 1)
        
        # Fit.
        self.regression = LinearRegression()
        self.regression.fit(X, y)

    def _predict(self) -> np.ndarray:
        """Predict the weight increases for the rest of the time period."""
        weight_predictions = self.regression.predict(
            self.df["weeks_from_start"].values.reshape(-1, 1)
        )
        weight_predictions = weight_predictions.reshape(1, -1)[0]

        return weight_predictions

    def get_df(self) -> pd.DataFrame:
        return self.df

    def write_to_csv(self):
        self.df.to_csv(f"output/bulk_tracker_output.csv")

    def plot(self):
        """
        Return plot of weight data overlayed on weight goal, with error bands 
        and end goal weight.
        """
        # Generate Series to plot.
        date_col = self.df.index
        weight_col = self.df["weight"]
        weekly_weight_goal = self.df["weekly_weight_goal"]
        lower_bound = self.df["lower_bound"]
        upper_bound = self.df["upper_bound"]
        # NOTE: predictions, may or may not plot
        # Predictions = self.df["weight_predictions"]
        
        # Begin plot, plot weight data, weekly goal, and error bands.
        sns.set()
        plt.plot(date_col, weekly_weight_goal, 'r--', label = "weight goal")
        plt.fill_between(date_col, lower_bound, upper_bound, color='r', alpha=0.2)
        plt.plot(date_col[-1], weekly_weight_goal[-1], 'r*', label = "end weight goal")
        # NOTE: predictions, may or may not plot
        # plt.plot(date_col, predictions, 'g-', label = "weight predictions")
        plt.plot(date_col, weight_col, 'b-', label = "measured weight")

        # Set labels, and axes.
        ax = plt.gca()
        ax.set_xticklabels(pd.to_datetime(self.df.index).date)
        plt.xlabel("Week")
        plt.ylabel("Weight")
        plt.title("Weight data")
        plt.legend()
        plt.show()
