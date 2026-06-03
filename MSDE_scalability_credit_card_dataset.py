import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from adbench.run_new import RunPipeline
from MSML_v10 import MSML  


csv_path = "creditcard.csv"
df = pd.read_csv(csv_path)


y = df["Class"].values.astype(int)   # 1 = fraud (anomaly)
X = df.drop(columns=["Class"]).values.astype(np.float32)

print("Dataset shape:", X.shape)
print("Anomaly ratio:", y.mean())


scaler = StandardScaler()
X = scaler.fit_transform(X)


dataset = {
    "X": X,
    "y": y
}


pipeline = RunPipeline(
    suffix="CreditCard",
    parallel="unsupervise",             
    realistic_synthetic_mode=None,      
    noise_type=None
)


results = pipeline.run(dataset=dataset, clf=MSML)
pd.DataFrame(results).to_csv('adbench/result/MSDE_credit_card.csv', index=False)

results2 = pipeline.run(dataset=dataset)
pd.DataFrame(results2).to_csv('adbench/result/benchmarks_credit_card.csv', index=False)


