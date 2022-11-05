import matplotlib.pyplot at plt
import pandas as pd

class viz(self):
    

    def candlesticks(self, df, volume=False):

        plt.figure(figsize=[7,7])
        # print(freqq)
        width = .8/(1400/int(freqq[:-1]))
        width2 = width/6.2

        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]

        up_color = 'green'
        down_color = 'red'

        plt.bar(up.index, up.close - up.open, width, bottom = up.open, color=up_color)
        plt.bar(up.index, up.high - up.close, width2, bottom = up.close, color=up_color)
        plt.bar(up.index, up.open - up.low, width2, bottom = up.low, color=up_color)

        plt.bar(down.index, down.open - down.close, width, bottom = down.close, color=down_color)
        plt.bar(down.index, down.high - down.open, width2, bottom = down.open, color=down_color)
        plt.bar(down.index, down.close - down.low, width2, bottom = down.low, color=down_color)

        plt.xticks(rotation=45, ha='right')
        plt.ylim(min(df['low'][df.low > 0]) - .2*(max(df['high'][df.high > 0]) - min(df['low'][df.low > 0])),
                max(df['high'][df.high > 0] + .1*(max(df['high'][df.high > 0]) - min(df['low'][df.low > 0]))))

        if volume == True:
            plt.twinx()
            plt.bar(up.index, up.volume, width*.8, color=up_color)
            plt.bar(down.index, down.volume, width*.8, color=down_color)
            plt.ylim(0, max(df['volume'])*10)
            plt.tick_params(right= False, labelright=False)

        return