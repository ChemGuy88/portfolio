import os
import re
import pandas as pd
from datetime import datetime as dt

datasetsPath = "data"

cpath = os.path.join(datasetsPath, "titanic.csv")
c=pd.read_csv(cpath)
 
test_labels = c
testpath = os.path.join(datasetsPath, "test.csv")
test = pd.read_csv(testpath)

for i, name in enumerate(test_labels['name']):
    if '"' in name:
        test_labels['name'][i] = re.sub('"', '', name)
        
for i, name in enumerate(test['Name']):
    if '"' in name:
        test['Name'][i] = re.sub('"', '', name)

survived = []

for name in test['Name']:
    survived.append(int(test_labels.loc[test_labels['name'] == name]['survived'].values[-1]))

examplePath = os.path.join(datasetsPath, "gender_submission.csv")
submission = pd.read_csv(examplePath)

submission['Survived'] = survived
timestamp = dt.now().strftime("%Y%m%d-%H%M%S")
submissionPath = os.path.join(datasetsPath, f"submission_{timestamp}.csv")
submission.to_csv(submissionPath, index=False)