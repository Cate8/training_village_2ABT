from session_plot import SessionPlot
import pandas as pd
import os



path = "/home/pi/village_projects/cate_task/data/sessions/RATON3/RATON3_S1_20250509_124927.csv"
width = 10
height = 8


path_plot = path[:-4] + ".pdf" 

session_plot = SessionPlot()

df = pd.read_csv(path, sep = ";")

fig = session_plot.create_plot(df, width, height)

fig.savefig(path_plot, format='pdf', bbox_inches='tight')





