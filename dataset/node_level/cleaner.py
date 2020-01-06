import pandas as pd
import numpy as np

tab = pd.read_csv("./facebook/target.csv")

new_tab = pd.DataFrame()


pages = tab["page_type"].values.tolist()

target = []
for p in pages:
    if p == "company":
        x = 0
    elif p == "tvshow":
        x = 1
    elif p == "politician":
        x = 2
    elif p == "government":
        x = 3
    target.append([x])

new_tab = pd.DataFrame(np.array(target).reshape(-1),columns = ["target"])
new_tab["id"] = tab["id"]
new_tab = new_tab[["id","target"]]
new_tab = new_tab.sort_values("id")

new_tab.to_csv("./facebook/new_target.csv",index=None)


#####################################


tab = pd.read_csv("./github/target.csv")

tab = pd.DataFrame(tab[["id","ml_target"]].values.tolist(), columns = ["id","target"])

new_tab = tab.sort_values("id")
new_tab.to_csv("./github/new_target.csv",index=None)


#####################################


tab = pd.read_csv("./wikipedia/target.csv")


pages = tab["target"].values.tolist()

z = np.quantile(pages, 0.715)
target = []
for p in pages:
    if p <z:
        x = 0
    else:
        x = 1
    target.append([x])

new_tab = pd.DataFrame(np.array(target).reshape(-1),columns = ["target"])
new_tab["id"] = tab["id"]
new_tab = new_tab[["id","target"]]
new_tab = new_tab.sort_values("id")
new_tab.to_csv("./wikipedia/new_target.csv",index=None)

############################################################


tab = pd.read_csv("./twitch/target.csv").values.tolist()

lines = []
for t in tab:
    if t[2] == True:
        line = [t[-1],1]
    else:
        line = [t[-1],0]
    lines.append(line)


new_tab = pd.DataFrame(lines,columns = ["id","target"])
new_tab = new_tab.sort_values("id")
new_tab.to_csv("./twitch/new_target.csv",index=None)




