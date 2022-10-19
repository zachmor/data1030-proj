import json
import os
import time
import quiverquant
from alpha_vantage.timeseries import TimeSeries
import pandas as pd


class Downloader:

    def __init__(self, alpha_token: str, quiver_token: str) -> None:
        self.quiver_client = quiverquant.quiver(quiver_token)
        self.alpha_client = TimeSeries(key=alpha_token, output_format='csv')

    def quiver_recent(self) -> None:

        try:
            os.mkdir("data")
        except FileExistsError:
            pass
        try:
            os.mkdir("data/quiver_recent")
        except FileExistsError:
            print("Overwriting...")
        methods = [method for method in dir(self.quiver_client) if method.startswith("__") is False]
        for method in methods:
            # print(method)
            f = getattr(self.quiver_client, method)
            if callable(f):
                try:
                    f().to_csv("data/quiver_recent/" + method + ".csv")
                    print("downloaded " + method)
                except NameError:
                    print("couldnt download " + method)

            else:
                print("NOT CALLABLE ", method)
        return

# Query quiver to download historical data for a specific ticker
    def quiver(self, ticker: str) -> None:

        try:
            os.mkdir("data")
        except FileExistsError:
            pass
        try:
            os.mkdir("data/quiver")
        except FileExistsError:
            print("Overwriting...")

        methods = [method for method in dir(self.quiver_client) if method.startswith("__") is False]
        for method in methods:
            # print(method)
            f = getattr(self.quiver_client, method)
            if callable(f):
                try:
                    f(ticker.upper()).to_csv("data/quiver/" + method + ".csv")
                    print("downloaded " + method)
                except NameError:
                    print("couldnt download " + method)
                except json.decoder.JSONDecodeError:
                    print("couldnt download " + method)
            else:
                print("NOT CALLABLE ", method)


# Using 30 min Intraday Time Series 
    def alpha(self, ticker: str) -> None:

        try:
            os.mkdir("data")
        except FileExistsError:
            pass
        try:
            os.mkdir("data/alpha")
        except FileExistsError:
            pass

        months = ["month" + str(i + 1) for i in range(12)]
        years = ["year1", "year2"]
        slice_names = [year + month for year in years for month in months]
        slices = []

        for i, s in enumerate(slice_names):
            # wrestling api call rate limits
            if i%5 == 0 and i != 0:
                time.sleep(60)
            data = self.alpha_client.get_intraday_extended(
                symbol=ticker.upper(),interval='1min', slice=s)
            columns = next(data[0])
            slices.append(pd.DataFrame(data=data[0], columns=columns))

        df = pd.concat(slices)
        df.to_csv("data/alpha/" + ticker.upper() + ".csv")

        # plt.title('Intraday Times Series for the MSFT stock (30 min)')
        # plt.show()
        # return