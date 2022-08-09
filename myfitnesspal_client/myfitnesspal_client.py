import myfitnesspal
import datetime
import pandas as pd
import os

class MyFitnessPalClient():

    def __init__(self, start_date, end_date=None, manual_download=False):
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.date.today()
        self.manual_download = manual_download

    def get_weight_data(self):
        weight_data = self.client.get_measurements('Weight', pd.to_datetime(self.start_date))

        return weight_data

    def get_weight_data_df(self):
        weight_data = self.get_weight_data()
        weight_list = [val for key, val in weight_data.items()]
        date_list = [key for key, val in weight_data.items()]
        weight_dict = {"weight": weight_list, "Date": date_list}
        weight_data_df = pd.DataFrame(weight_dict)
        weight_data_df.sort_values(by="Date", ascending=True, inplace=True)
        weight_data_df.set_index("Date", inplace=True)

        return weight_data_df

    def get_nutrition_data_df(self):
        
        date_list = pd.date_range(start=self.start_date, end=self.end_date)
        calorie_list = [self.client.get_date(date).totals.get("calories") for date in date_list]

        nutrition_data_dict = {"calories": calorie_list, "Date": date_list}
        nutrition_data_df = pd.DataFrame(nutrition_data_dict)
        nutrition_data_df.set_index("Date", inplace=True)

        return nutrition_data_df

    def download_format_data(self, filename):
        self.client = myfitnesspal.Client(username="vdomingu3z", password="Sasuke556!")
        weight_data_df = self.get_weight_data_df()
        nutrition_data_df = self.get_nutrition_data_df()
        mfp_df = pd.concat([weight_data_df, nutrition_data_df], axis=1)
        mfp_df.index = pd.to_datetime(mfp_df.index)
        mfp_df.to_csv(filename)

        return mfp_df

    def get_myfitnesspal_df(self):

        filename = f"weight_data/my_fitnesspal_data_from_{self.start_date}.csv"

        if not os.path.exists(filename) or self.manual_download:
            mfp_df = self.download_format_data(filename)
        
        else:
            mfp_df = pd.read_csv(filename)
            mfp_df["Date"] = pd.to_datetime(mfp_df["Date"])
            mfp_df.set_index("Date", inplace=True)

        return mfp_df

if __name__ == "__main__":
    START_DATE = datetime.datetime(2022, 7, 1)
    manual_download = True


    mfpc = MyFitnessPalClient(start_date=START_DATE, manual_download=manual_download)
    mfpc.download_format_data('my_file.csv')
