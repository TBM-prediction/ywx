from utils.imports import *

def set_font(label, fname='/usr/share/fonts/wqy-microhei/wqy-microhei.ttc'):
    # use chinese fonts
    from matplotlib import font_manager as fm
    prop = fm.FontProperties(fname=fname)
    plt.setp(label, fontproperties=prop)


def plots(df_raw, cols=3, unit_figsize=(8, 3), ax=None, #fname='/System/Library/Fonts/PingFang.ttc'):
                                     fname='/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',
                                     title=None):
    num_plots = df_raw.shape[1] #- 2 # omit index and timestamp
    rows = num_plots // cols + 1
    figsize = (unit_figsize[0] * cols, unit_figsize[1] * rows)

    ax = ax[:num_plots] if ax is not None else ax
    axes = df_raw.plot(subplots=True, figsize=figsize, 
            layout=(rows, cols), ax=ax, legend=ax is None, title=title)
            #fontsize=0,layout=(rows, cols), ax=ax, legend=False)

    if ax is None:
    #if False: # no legend
        for ax in axes.flatten():
            legend = ax.legend()
            set_font(legend.texts, fname=fname)
            xlabel = ax.xaxis.label
            set_font(xlabel, fname=fname)

    plt.tight_layout()
    return axes

def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)
