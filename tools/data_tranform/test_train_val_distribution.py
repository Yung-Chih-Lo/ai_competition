import  pandas as pd

train_df = pd.read_hdf("datasets/origin/train.h5", key="data")
val_df = pd.read_hdf("datasets/origin/val.h5", key="data")

print("Train 飆股分佈:")
print(train_df['飆股'].value_counts())
print("Val 飆股分佈:")
print(val_df['飆股'].value_counts())
