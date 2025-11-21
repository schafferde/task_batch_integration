import sys
import pandas as pd
#Stacks scib-metrics benchmarking DFs
#Args: output_df.pkl input_df.1.pkl input_df.2.pkl input_df.3.pkl...

dfs = []
for name in sys.argv[2:]:
    dfs.append(pd.read_pickle(name))
for df in dfs[:-1]:
    df.drop(df.tail(1).index,inplace=True)
df2 = pd.concat(dfs)
print(df2.shape)
df2.to_pickle(sys.argv[1])