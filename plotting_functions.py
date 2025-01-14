import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_side_correct_performance(df: pd.DataFrame, ax: plt.Axes) -> plt.Axes:
    ax.clear()
    # select only the last 100 trials
    df = df.tail(100)
    sns.scatterplot(data=df, x="trial", y="trial_type", hue="correct", ax=ax)
    # plot the mean of the last 5 trials
    ax.plot(pd.Series([int(x) for x in df.correct]).rolling(5).mean(), "r")

    return ax

if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "trial": range(100),
            "trial_type": ["left" if x % 2 == 0 else "right" for x in range(100)],
            "correct": [True if x % 3 == 0 else False for x in range(100)],
        }
    )
    fig, ax = plt.subplots()
    plot_side_correct_performance(df, ax)
    plt.show()