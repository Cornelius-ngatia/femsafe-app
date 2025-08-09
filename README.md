# Predictive Femicide Risk Detection in Kenya

## Problem Statement
Femicide—the gender-based killing of women—is a growing concern in Kenya, with over 150 cases reported in 2023 alone. These deaths are often the culmination of repeated gender-based violence (GBV), emotional abuse, and systemic failures to intervene. Traditional response mechanisms are reactive, often triggered too late.

There is an urgent need for a data-driven, preventative approach that can flag high-risk cases early and guide timely interventions from law enforcement, social workers, or NGOs.


## Stakeholders
The success of a preventative femicide project relies on collaboration across sectors. Below are the key stakeholders and their roles:

- NGOs and Civil Society Organizations
e.g., FIDA Kenya, Wangu Kanja Foundation
Support survivors, use risk assessment tools for early interventions, provide shelters, and report case data.

- Law Enforcement Agencies
Use predictive tools to identify high-risk individuals or regions and prioritize protective action.

- Healthcare Providers
Flag victims with recurring injuries, mental health issues, or delayed reporting patterns.

- Judiciary and Legal Practitioners
Incorporate risk indicators into protection orders, court rulings, or bail/parole decisions.

- Government and Policy Makers
Allocate resources more effectively, design GBV prevention policies based on insights.

- Survivors and Local Communities
Receive better protection and support through proactive systems designed to detect risk.

- Data Scientists and Researchers
Develop, test, and maintain ethical machine learning models for prevention and triage.

## Objectives
1. To develop an AI-powered system that analyzes user-reported messages and behavior patterns to detect early signs of gender-based violence and potential femicide risk.
2. To provide real-time safety features such as a panic button, emergency contact alerts, and access to support services including counseling, shelters, and legal aid
3. To raise awareness and educate users on recognizing abuse, understanding their rights, and building safety plans through localized, culturally relevant content


```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as snS
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')

import warnings
warnings.filterwarnings('ignore')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\HP\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    

#### FIRST DATASET


```python
# Load the dataset
df_1 = pd.read_excel("kenya-femicide-data_2016_dec2023 (3).xlsx")

# Show the first few rows
df_1.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>title</th>
      <th>author_name</th>
      <th>medium</th>
      <th>country_name</th>
      <th>text</th>
      <th>published_date</th>
      <th>type of murder</th>
      <th>name of victim</th>
      <th>Age</th>
      <th>...</th>
      <th>Type of femicide</th>
      <th>Murder Scene</th>
      <th>Mode of killing</th>
      <th>Circumstance</th>
      <th>Status on article date</th>
      <th>Court date (first appearance)</th>
      <th>verdict date</th>
      <th>Verdict</th>
      <th>Years of sentence</th>
      <th>Days to verdict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>http://www.standardmedia.co.ke/article/2000218...</td>
      <td>Meru man who killed drunk wife sentenced to hang</td>
      <td>Lydiah Nyawira</td>
      <td>Standard Digital (Kenya)</td>
      <td>Kenya</td>
      <td>On August 12, 2005 at around 7pm, Joyce Gacher...</td>
      <td>2016-10-05</td>
      <td>Femicide</td>
      <td>Dorcas Kaguri</td>
      <td>Unknown</td>
      <td>...</td>
      <td>Intimate</td>
      <td>home</td>
      <td>blunt force</td>
      <td>Argument</td>
      <td>Unknown</td>
      <td>2005-08-13 00:00:00</td>
      <td>2016-06-24 00:00:00</td>
      <td>Guilty</td>
      <td>Death</td>
      <td>4061.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http://www.standardmedia.co.ke/article/2001300...</td>
      <td>Court: Kakamega man guilty of raping, killing ...</td>
      <td>Jack Murima</td>
      <td>Standard Digital (Kenya)</td>
      <td>Kenya</td>
      <td>Jack Murima - Sun, 28. October 2018 12:54 PM -...</td>
      <td>2018-10-29</td>
      <td>Femicide</td>
      <td>Selina Ikambi Anasi</td>
      <td>79</td>
      <td>...</td>
      <td>Non-intimate</td>
      <td>Public Space</td>
      <td>hacked</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>2018-10-28 00:00:00</td>
      <td>guilty</td>
      <td>awaiting ruling</td>
      <td>2817.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>http://www.standardmedia.co.ke/article/2001241...</td>
      <td>My wife 'died from fall in fight over volume o...</td>
      <td>Kamau Muthoni</td>
      <td>Standard Digital (Kenya)</td>
      <td>Kenya</td>
      <td>Former journalist Moses Dola Otieno yesterday ...</td>
      <td>2017-05-25</td>
      <td>Femicide</td>
      <td>Wambui Kabiru</td>
      <td>Unknown</td>
      <td>...</td>
      <td>Intimate</td>
      <td>home</td>
      <td>blunt force</td>
      <td>Argument</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>2018-10-05 00:00:00</td>
      <td>guilty</td>
      <td>10</td>
      <td>2714.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>http://www.nation.co.ke/news/Kabogo-cleared-Me...</td>
      <td>Kabogo cleared in varsity student's murder case</td>
      <td>ELISHA OTIENO</td>
      <td>Daily Nation (Kenya)</td>
      <td>Kenya</td>
      <td>Kiambu Governor Kabogo and five others adverse...</td>
      <td>2016-10-07</td>
      <td>Femicide</td>
      <td>Mercy Keino</td>
      <td>Unknown</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Public Space</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>2016-10-07 00:00:00</td>
      <td>Not Guilty</td>
      <td>Freed</td>
      <td>1939.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>http://www.mediamaxnetwork.co.ke/news/pastor-t...</td>
      <td>Pastor to serve 25 years in jail for defiling,...</td>
      <td>People Daily</td>
      <td>MediaMax Network</td>
      <td>Kenya</td>
      <td>Musa Wekesa A pastor from Kitale has been sent...</td>
      <td>2019-05-25</td>
      <td>Femicide</td>
      <td>Scholastica Mmbihi</td>
      <td>Unknown</td>
      <td>...</td>
      <td>Unknown</td>
      <td>home</td>
      <td>stabbed</td>
      <td>Argument</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>2019-05-25 00:00:00</td>
      <td>guilty</td>
      <td>25</td>
      <td>2879.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
df_1.columns
```




    Index(['url', 'title', 'author_name', 'medium', 'country_name', 'text',
           'published_date', 'type of murder', 'name of victim', 'Age',
           'date of murder', 'Location', 'name of suspect', 'suspect relationship',
           'Type of femicide', 'Murder Scene', 'Mode of killing', 'Circumstance',
           'Status on article date', 'Court date (first appearance)',
           'verdict date', 'Verdict', 'Years of sentence', 'Days to verdict'],
          dtype='object')




```python
# Basic info: columns, non-null counts, data types
df_1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 507 entries, 0 to 506
    Data columns (total 24 columns):
     #   Column                         Non-Null Count  Dtype         
    ---  ------                         --------------  -----         
     0   url                            507 non-null    object        
     1   title                          479 non-null    object        
     2   author_name                    473 non-null    object        
     3   medium                         495 non-null    object        
     4   country_name                   492 non-null    object        
     5   text                           468 non-null    object        
     6   published_date                 479 non-null    datetime64[ns]
     7   type of murder                 507 non-null    object        
     8   name of victim                 507 non-null    object        
     9   Age                            507 non-null    object        
     10  date of murder                 507 non-null    object        
     11  Location                       505 non-null    object        
     12  name of suspect                468 non-null    object        
     13  suspect relationship           505 non-null    object        
     14  Type of femicide               448 non-null    object        
     15  Murder Scene                   507 non-null    object        
     16  Mode of killing                507 non-null    object        
     17  Circumstance                   438 non-null    object        
     18  Status on article date         436 non-null    object        
     19  Court date (first appearance)  408 non-null    object        
     20  verdict date                   507 non-null    object        
     21  Verdict                        406 non-null    object        
     22  Years of sentence              405 non-null    object        
     23  Days to verdict                36 non-null     float64       
    dtypes: datetime64[ns](1), float64(1), object(22)
    memory usage: 95.2+ KB
    


```python
df_1.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>title</th>
      <th>author_name</th>
      <th>medium</th>
      <th>country_name</th>
      <th>text</th>
      <th>published_date</th>
      <th>type of murder</th>
      <th>name of victim</th>
      <th>Age</th>
      <th>...</th>
      <th>Type of femicide</th>
      <th>Murder Scene</th>
      <th>Mode of killing</th>
      <th>Circumstance</th>
      <th>Status on article date</th>
      <th>Court date (first appearance)</th>
      <th>verdict date</th>
      <th>Verdict</th>
      <th>Years of sentence</th>
      <th>Days to verdict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>507</td>
      <td>479</td>
      <td>473</td>
      <td>495</td>
      <td>492</td>
      <td>468</td>
      <td>479</td>
      <td>507</td>
      <td>507</td>
      <td>507</td>
      <td>...</td>
      <td>448</td>
      <td>507</td>
      <td>507</td>
      <td>438</td>
      <td>436</td>
      <td>408</td>
      <td>507</td>
      <td>406</td>
      <td>405</td>
      <td>36.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>507</td>
      <td>476</td>
      <td>285</td>
      <td>26</td>
      <td>1</td>
      <td>468</td>
      <td>NaN</td>
      <td>2</td>
      <td>399</td>
      <td>65</td>
      <td>...</td>
      <td>7</td>
      <td>10</td>
      <td>13</td>
      <td>48</td>
      <td>15</td>
      <td>67</td>
      <td>37</td>
      <td>6</td>
      <td>20</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>http://www.standardmedia.co.ke/article/2000218...</td>
      <td>Kisumu man arrested for killing wife after dom...</td>
      <td>Cyrus Ombati</td>
      <td>The Star (Kenya)</td>
      <td>Kenya</td>
      <td>On August 12, 2005 at around 7pm, Joyce Gacher...</td>
      <td>NaN</td>
      <td>Femicide</td>
      <td>Unnamed</td>
      <td>Unknown</td>
      <td>...</td>
      <td>Intimate</td>
      <td>home</td>
      <td>stabbed</td>
      <td>Argument</td>
      <td>under investigation</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>2</td>
      <td>14</td>
      <td>176</td>
      <td>492</td>
      <td>1</td>
      <td>NaN</td>
      <td>493</td>
      <td>103</td>
      <td>195</td>
      <td>...</td>
      <td>311</td>
      <td>256</td>
      <td>143</td>
      <td>158</td>
      <td>274</td>
      <td>341</td>
      <td>471</td>
      <td>368</td>
      <td>369</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-04-23 07:24:55.615866624</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1688.555556</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-01-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-84.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-07-02 12:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>672.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-04-27 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1726.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2022-01-09 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2699.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2023-12-06 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4061.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1135.545343</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 24 columns</p>
</div>




```python
# Check for missing values
df_1.isnull().sum()
```




    url                                0
    title                             28
    author_name                       34
    medium                            12
    country_name                      15
    text                              39
    published_date                    28
    type of murder                     0
    name of victim                     0
    Age                                0
    date of murder                     0
    Location                           2
    name of suspect                   39
    suspect relationship               2
    Type of femicide                  59
    Murder Scene                       0
    Mode of killing                    0
    Circumstance                      69
    Status on article date            71
    Court date (first appearance)     99
    verdict date                       0
    Verdict                          101
    Years of sentence                102
    Days to verdict                  471
    dtype: int64




```python
# Check all columns for literal string "Unknown"
unknown_all = df_1.apply(lambda col: col.astype(str).str.lower().eq('unknown').sum())
unknown_all = unknown_all[unknown_all > 0].sort_values(ascending=False)
print(unknown_all)
```

    verdict date                     471
    Years of sentence                369
    Verdict                          368
    Court date (first appearance)    341
    Age                              195
    name of suspect                  154
    Circumstance                     149
    Mode of killing                   66
    Status on article date            59
    Type of femicide                  50
    Murder Scene                      38
    date of murder                    14
    author_name                        9
    name of victim                     3
    Location                           2
    dtype: int64
    


```python
# Copy the original DataFrame
df_1_cleaned = df_1.copy()

# Replace 'unknown' or '#VALUE!' with NaN
df_1_cleaned.replace(
    r'(?i)^\s*(unknown|#VALUE!)\s*$',
    np.nan,
    regex=True,
    inplace=True)

# Let pandas infer nullable dtypes
df_1_cleaned = df_1_cleaned.convert_dtypes()

# Revert StringDtype to object (so later fillna('Unknown') works)
str_ext = df_1_cleaned.select_dtypes(include='string').columns
df_1_cleaned[str_ext] = df_1_cleaned[str_ext].astype('object')

# Revert numeric extension types to numpy built-ins

int_cols = df_1_cleaned.select_dtypes(include='int').columns
float_cols = df_1_cleaned.select_dtypes(include='float').columns
bool_cols = df_1_cleaned.select_dtypes(include='bool').columns



for col in int_cols:
    s = df_1_cleaned[col]
    df_1_cleaned[col] = s.astype('int64') if not s.isna().any() else s.astype('float64')

df_1_cleaned[float_cols] = df_1_cleaned[float_cols].astype('float64')
df_1_cleaned[bool_cols]  = df_1_cleaned[bool_cols].astype('bool')

# Convert verdict date to datetime
df_1_cleaned['verdict date'] = pd.to_datetime(
    df_1_cleaned['verdict date'],
    dayfirst=True,
    errors='coerce'
)


# check if NaNs still present
print("Remaining missing values:", df_1_cleaned.isna().sum())
print("Current dtypes:", df_1_cleaned.dtypes)
```

    Remaining missing values: url                                0
    title                             28
    author_name                       43
    medium                            12
    country_name                      15
    text                              39
    published_date                    28
    type of murder                     0
    name of victim                     3
    Age                              195
    date of murder                    14
    Location                           4
    name of suspect                  193
    suspect relationship               2
    Type of femicide                 109
    Murder Scene                      38
    Mode of killing                   66
    Circumstance                     218
    Status on article date           130
    Court date (first appearance)    440
    verdict date                     471
    Verdict                          469
    Years of sentence                471
    Days to verdict                  471
    dtype: int64
    Current dtypes: url                                      object
    title                                    object
    author_name                              object
    medium                                   object
    country_name                             object
    text                                     object
    published_date                   datetime64[ns]
    type of murder                           object
    name of victim                           object
    Age                                     float64
    date of murder                   datetime64[ns]
    Location                                 object
    name of suspect                          object
    suspect relationship                     object
    Type of femicide                         object
    Murder Scene                             object
    Mode of killing                          object
    Circumstance                             object
    Status on article date                   object
    Court date (first appearance)            object
    verdict date                     datetime64[ns]
    Verdict                                  object
    Years of sentence                        object
    Days to verdict                         float64
    dtype: object
    


```python
# Identify categorical and numeric columns
categorical_cols = df_1_cleaned.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df_1_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical columns:", categorical_cols)
print()
print("nNumeric columns:", numeric_cols)

```

    Categorical columns: ['url', 'title', 'author_name', 'medium', 'country_name', 'text', 'type of murder', 'name of victim', 'Location', 'name of suspect', 'suspect relationship', 'Type of femicide', 'Murder Scene', 'Mode of killing', 'Circumstance', 'Status on article date', 'Court date (first appearance)', 'Verdict', 'Years of sentence']
    
    nNumeric columns: ['Age', 'Days to verdict']
    


```python
df_1_cleaned['Age'] = df_1_cleaned['Age'].fillna(df_1_cleaned['Age'].median())

```


```python
df_1_cleaned['date of murder'] = pd.to_datetime(df_1_cleaned['date of murder'], errors='coerce')
df_1_cleaned['published_date'] = pd.to_datetime(df_1_cleaned['published_date'], errors='coerce')

```


```python
df_1_cleaned['published_date'] = df_1_cleaned['published_date'].fillna(pd.Timestamp('1900-01-01'))
df_1_cleaned['date of murder'] = df_1_cleaned['date of murder'].fillna(pd.Timestamp('1900-01-01'))

```


```python
df_1_cleaned['name of victim'].duplicated().sum()
```




    108




```python
#  Start from a clean copy
df__1_cleaned = df_1.copy()

# Regex to catch all placeholder strings
placeholder_pattern = r'(?i)^\s*(?:nan|none|null|unknown|#value!)\s*$'

# Identify all text-like columns
cat_cols = df_1_cleaned.select_dtypes(include=['object','string','category']).columns

# Force them to object dtype
df_1_cleaned[cat_cols] = df_1_cleaned[cat_cols].astype('object')

# Replace placeholders with real NaN
df_1_cleaned[cat_cols] = (
    df_1_cleaned[cat_cols]
    .replace(placeholder_pattern, np.nan, regex=True)
)

# Fill NaNs in categorical columns with "Unknown"
df_1_cleaned[cat_cols] = df_1_cleaned[cat_cols].fillna('Unknown')

# Convert Age to numeric & median-impute
df_1_cleaned['Age'] = pd.to_numeric(df_1_cleaned['Age'], errors='coerce')
df_1_cleaned['Age'] = df_1_cleaned['Age'].fillna(df_1_cleaned['Age'].median())

# Identify any other numeric columns
other_num_cols = [
    c for c in df_1_cleaned.select_dtypes(include=[np.number]).columns
    if c != 'Age'
]

# Coerce and median-impute the rest
df_1_cleaned[other_num_cols] = (
    df_1_cleaned[other_num_cols]
    .apply(lambda s: pd.to_numeric(s, errors='coerce'))
    .fillna(df_1_cleaned[other_num_cols].median())
)

# missing-value check
print("Categorical missing:", df_1_cleaned[cat_cols].isna().sum().sum())
print("Numeric missing:", df_1_cleaned[['Age'] + other_num_cols].isna().sum().sum())

```

    Categorical missing: 0
    Numeric missing: 0
    


```python
# Exclude these columns
exclude_cols = ["url", "text", "name of victim"]

#Object columns
cols = [
    c for c in df_1_cleaned.select_dtypes(include=["object"]).columns
    if c not in exclude_cols
]

# Normalize, strip, lower, then Title-case
for col in cols:
    df_1_cleaned[col] = (
        df_1_cleaned[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.title())


print(df_1_cleaned[cols].head())
print(df_1_cleaned[exclude_cols].head())
```

                                                   title     author_name  \
    0   Meru Man Who Killed Drunk Wife Sentenced To Hang  Lydiah Nyawira   
    1  Court: Kakamega Man Guilty Of Raping, Killing ...     Jack Murima   
    2  My Wife 'Died From Fall In Fight Over Volume O...   Kamau Muthoni   
    3    Kabogo Cleared In Varsity Student'S Murder Case   Elisha Otieno   
    4  Pastor To Serve 25 Years In Jail For Defiling,...    People Daily   
    
                         medium country_name type of murder  \
    0  Standard Digital (Kenya)        Kenya       Femicide   
    1  Standard Digital (Kenya)        Kenya       Femicide   
    2  Standard Digital (Kenya)        Kenya       Femicide   
    3      Daily Nation (Kenya)        Kenya       Femicide   
    4          Mediamax Network        Kenya       Femicide   
    
                              Location         name of suspect  \
    0    Gaintume Village, Meru County            Julius Koome   
    1  Mituri Village, Kakamega County  Wilson Masiko Mungasia   
    2     Umoja Estate, Nairobi County       Moses Dola Otieno   
    3      Waiyaki Way, Nairobi County          William Kabogo   
    4       Kitale, Trans Nzoia County        Pst. Musa Wekesa   
    
                suspect relationship Type of femicide  Murder Scene  \
    0                        Husband         Intimate          Home   
    1                  Family Member     Non-Intimate  Public Space   
    2                        Husband         Intimate          Home   
    3  Stranger/Unknown Relationship          Unknown  Public Space   
    4  Stranger/Unknown Relationship          Unknown          Home   
    
      Mode of killing Circumstance Status on article date  \
    0     Blunt Force     Argument                Unknown   
    1          Hacked      Unknown                Unknown   
    2     Blunt Force     Argument                Unknown   
    3         Unknown      Unknown                Unknown   
    4         Stabbed     Argument                Unknown   
    
      Court date (first appearance)     Verdict Years of sentence  
    0           2005-08-13 00:00:00      Guilty             Death  
    1                       Unknown      Guilty   Awaiting Ruling  
    2                       Unknown      Guilty                10  
    3                       Unknown  Not Guilty             Freed  
    4                       Unknown      Guilty                25  
                                                     url  \
    0  http://www.standardmedia.co.ke/article/2000218...   
    1  http://www.standardmedia.co.ke/article/2001300...   
    2  http://www.standardmedia.co.ke/article/2001241...   
    3  http://www.nation.co.ke/news/Kabogo-cleared-Me...   
    4  http://www.mediamaxnetwork.co.ke/news/pastor-t...   
    
                                                    text       name of victim  
    0  On August 12, 2005 at around 7pm, Joyce Gacher...        Dorcas Kaguri  
    1  Jack Murima - Sun, 28. October 2018 12:54 PM -...  Selina Ikambi Anasi  
    2  Former journalist Moses Dola Otieno yesterday ...        Wambui Kabiru  
    3  Kiambu Governor Kabogo and five others adverse...          Mercy Keino  
    4  Musa Wekesa A pastor from Kitale has been sent...   Scholastica Mmbihi  
    


```python
import unicodedata
from rapidfuzz import process, fuzz

# Normalize valid counties
valid_counties = [
    'Baringo', 'Bomet', 'Bungoma', 'Busia', 'Elgeyo Marakwet', 'Embu', 'Garissa',
    'Homa Bay', 'Isiolo', 'Kajiado', 'Kakamega', 'Kericho', 'Kiambu', 'Kilifi',
    'Kirinyaga', 'Kisii', 'Kisumu', 'Kitui', 'Kwale', 'Laikipia', 'Lamu',
    'Machakos', 'Makueni', 'Mandera', 'Marsabit', 'Meru', 'Migori', 'Mombasa',
    'Murang’a', 'Nairobi', 'Nakuru', 'Nandi', 'Narok', 'Nyamira', 'Nyandarua',
    'Nyeri', 'Samburu', 'Siaya', 'Taita Taveta', 'Tana River', 'Tharaka Nithi',
    'Trans Nzoia', 'Turkana', 'Uasin Gishu', 'Vihiga', 'Wajir', 'West Pokot'
]
normalized_counties = [unicodedata.normalize('NFKD', c).encode('ascii', 'ignore').decode().lower() for c in valid_counties]

# Helpers
def normalize_text(text):
    if pd.isnull(text): return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode()
    return re.sub(r"[’'`]", "", text).lower().strip()

def extract_county_from_text(text):
    match = re.search(r'([A-Za-z’\'\-\s]+?)\s+County', str(text), re.IGNORECASE)
    if match:
        return normalize_text(match.group(1))
    return None
```


```python
# Strip, Title case, and try to resolve county
df_1_cleaned['Location'] = df_1_cleaned['Location'].astype(str).str.strip().str.title()

def resolve_county(row):
    location = row.get('Location', '')
    raw_county = row.get('County', '')

    # Direct clean match
    cleaned = normalize_text(raw_county)
    if cleaned in normalized_counties:
        return valid_counties[normalized_counties.index(cleaned)]

    # Pattern match like "X County"
    extracted = extract_county_from_text(location)
    if extracted:
        guess, score, _ = process.extractOne(extracted, normalized_counties, scorer=fuzz.ratio)
        if score > 85:
            return valid_counties[normalized_counties.index(guess)]

    # Fuzzy match full location text
    location_clean = normalize_text(location)
    guess, score, _ = process.extractOne(location_clean, normalized_counties, scorer=fuzz.ratio)
    if score > 85:
        return valid_counties[normalized_counties.index(guess)]

    return None

df_1_cleaned['County'] = df_1_cleaned.apply(resolve_county, axis=1)

```


```python
unmatched_locations = df_1_cleaned[df_1_cleaned['County'].isna()]['Location'].dropna().unique().tolist()
print(unmatched_locations)

```

    ['Unknown', 'Kamiti Corner In Kasarani', 'Highway Complex Lodge, Migory County', 'Nyayo Estate, Embakasi, Nairobi', 'Umoja Iii Estate', 'Freehold Estate, Nakuru', 'Muthaiga Estate', 'Muthure, Kikuyu Constituency', 'Kahawa Wendani', 'Karuku, Kandara\nSubcounty, Muranga', 'Kiganjo Estate In Thika.', 'Kahawa Sukari', 'Tom Mboya Street, Nairobi', 'Egerton University, Njokerio Area, Njoro', "Maragua Ridge, Murang'A South", 'Eldoret Townuasin Gishu County', 'Biasumu Village, Migory County', 'Mbuyuni Village In Likoni, Mombasa', 'Bundo In Bogichora, Nyamira South', 'Sinderma Village', 'Lanet, Nakuru', 'Kitengela', 'Bendera Area, West Pokot', 'Lwala Sub Location, Gwassi East Location In Suba Sub-County', 'Naishi Village, Muthira Sub-Location, Njoro Constituency In Nakuru', 'Kasarani', 'Kangemi, Nairobi, County', 'Kosoywa Estate, Near Nandi Hills Town, Nnadi County', 'Kikumbo Village, Kalama Sub-County.', 'Lari, Limuru', 'Mtongwe, Mombasa', 'Tembwo Trading Centre,Sotik Sub County, Bomet', 'Tushauriane, Kayole, Nairobi', 'Kabiru Village, Nanyuki', 'Maili Tisa Village In Loitokitok', 'Rabuor, Kisumu', 'Mwanzai, Sagwa, Kaloleni In Kilifi', 'Bombolulu, Mombasa', 'Ochieng B Village In Ligega Sub-Location, Ugenya Sub-County', 'Kiprambu In Chepkumia, Kapsabet', 'Thiba Village In Kirinyaga, County', 'Nyakoingwana Village, Endebess Sub County, Trans Nzoia County', "Mugumoini Chief'S Camp, Langata, Nairobi", 'Zakayos Estate, Nakuru', 'Kasarani, Nairobi', 'Roadside In Jamhuri Estate, Nairobi.', 'Kitui Town', 'Mitunguu Market, Imenti South', 'Mitikenda Estate On The Outskirts Of Ruiru', 'Bush Near Maasai Mara University, Narok', 'Burgei Village In Kipkelion East Constituency', 'Mukuyu, Muranga', 'Rwamuthambi River In Kagio Town', 'Gathugu Village In Mathira.', 'Kamulu', 'Kibolo Village, Turbo, Uasin Gishu', 'Buruburu, Nairobi', 'Awendo Town-Siruti Road', 'Kiaragana, Embu.', 'Karagita Area, Naivasha', 'Kanyakine, Imenti South', 'Riverside, Kiambu', 'Serem Area Of Vihiga County', 'Wamunyiri Village, Kanduyi Constituency', 'Kiwanjani Area, Kajiado', 'Ematunzi Village, Em’Mutsa Sub-Location In Vihiga County', 'Kiima Village In Kamuwongo Division Of Kyuso Sub County', 'Mathare Estate,Maragua Town', 'Chebon Location., West Pokot', 'Kamahohu In Tetu', 'Kasambara Village, Gilgil Sub County', 'Igoki Village, Imenti North', 'Kodwar B Village, Kodumo East Sub-Location In Rachuonyo East Sub-County', 'Kariadudu, Baba Ndogo Nairobi', "Gaturo Village, Murang'A", 'James Finlays Tea Company Workshop Estate, Kericho', 'Railways Police Lines In Eldoret', 'Plainsview In South B', 'Bandani Village In Kisumu.']
    


```python
# If still missing, try extracting from comma-separated part
df_1_cleaned['County'] = df_1_cleaned['County'].fillna(
    df_1_cleaned['Location'].str.split(',').str[-1].str.strip().str.title()
)

# Town-to-county mapping
town_to_county = {
    'Kitengela': 'Kajiado', 'Kasarani': 'Nairobi', 'Muchatha': 'Kiambu',
    'Santonia Court': 'Nairobi', 'Rukanga Village': 'Kirinyaga', 'Iten Town': 'Elgeyo Marakwet',
    'Eldoret Town': 'Uasin Gishu', 'Naivasha': 'Nakuru', 'Gilgil': 'Nakuru',
    'Buruburu Phase 5': 'Nairobi', 'Umoja Estate': 'Nairobi', 'Kibera Slums': 'Nairobi',
    'Ongata Rongai': 'Kajiado', 'Navakholo': 'Kakamega', 'Kitale': 'Trans Nzoia',
    'Kisii': 'Kisii'
}

#  for unmatched  cases
manual_fixes = {
    'Tembwo Trading Centre,Sotik Sub County, Bomet': 'Bomet',
    'Biasumu Village, Migory County': 'Migori',
    'Kosoywa Estate, Near Nandi Hills Town, Nnadi County': 'Nandi',
    'Sinderma Village': 'West Pokot',
    'Kamulu': 'Nairobi',
    'Karagita Area, Naivasha': 'Nakuru',
    'Railways Police Lines In Eldoret': 'Uasin Gishu',
    'Bandani Village In Kisumu.': 'Kisumu'}
 

# Apply town & manual mapping
def final_fix(row):
    loc = row['Location'].split(',')[0].strip().title()
    return manual_fixes.get(row['Location'], town_to_county.get(loc, row['County']))

df_1_cleaned['County'] = df_1_cleaned.apply(final_fix, axis=1)

# Normalize minor spelling inconsistencies
county_fix_map = {
    'Muranga': 'Murang’a', 'Muranga County': 'Murang’a', 'Murang\'A': 'Murang’a',
    'Unknown': None, 'Nan': None, 'X': None
}
df_1_cleaned['County'] = df_1_cleaned['County'].replace(county_fix_map).str.strip().str.title()

```


```python
#  Fill NaN County using last part of Location
df_1_cleaned['County'] = df_1_cleaned['County'].fillna(
    df_1_cleaned['Location'].str.split(',').str[-1].str.strip().str.title()
)

# Town to County mapping
town_to_county = {
    'Kitengela': 'Kajiado', 'Kasarani': 'Nairobi', 'Muchatha': 'Kiambu',
    'Santonia Court': 'Nairobi', 'Rukanga Village': 'Kirinyaga', 'Iten Town': 'Elgeyo Marakwet',
    'Eldoret Town': 'Uasin Gishu', 'Naivasha': 'Nakuru', 'Gilgil': 'Nakuru',
    'Buruburu Phase 5': 'Nairobi', 'Umoja Estate': 'Nairobi', 'Kibera Slums': 'Nairobi',
    'Ongata Rongai': 'Kajiado', 'Navakholo': 'Kakamega', 'Kitale': 'Trans Nzoia',
    'Kisii': 'Kisii'}

# Manual override mapping for edge cases
manual_fixes = {
    'Tembwo Trading Centre,Sotik Sub County, Bomet': 'Bomet',
    'Biasumu Village, Migory County': 'Migori',
    'Kosoywa Estate, Near Nandi Hills Town, Nnadi County': 'Nandi',
    'Sinderma Village': 'West Pokot',
    'Kamulu': 'Machakos',
    'Karagita Area, Naivasha': 'Nakuru',
    'Railways Police Lines In Eldoret': 'Uasin Gishu',
    'Bandani Village In Kisumu.': 'Kisumu',
    'Kalama Sub-County.': 'Machakos',
    'Limuru': 'Kiambu',
    'Sotik Sub': 'Bomet',
    'Nanyuki': 'Laikipia',
    'Maili Tisa Village In Loitokitok': 'Kajiado',
    'Kaloleni In Kilifi': 'Kilifi',
    'Ugenya Sub-County': 'Siaya',
    'Kapsabet': 'Nandi',
    'Endebess Sub': 'Trans Nzoia',
    'Kitui Town': 'Kitui',
    'Imenti South': 'Meru',
    'Mitikenda Estate On The Outskirts Of Ruiru': 'Kiambu',
    'Burgei Village In Kipkelion East Constituency': 'Kericho',
    'Rwamuthambi River In Kagio Town': 'Nyeri',
    'Gathugu Village In Mathira.': 'Nyeri',
    'Awendo Town-Siruti Road': 'Migori',
    'Embu.': 'Embu',
    'Naivasha': 'Nakuru',
    'Serem Area Of Vihiga': 'Vihiga',
    'Kanduyi Constituency': 'Bungoma',
    'Location In Vihiga': 'Vihiga',
    'Kiima Village In Kamuwongo Division Of Kyuso Sub': 'Kitui',
    'Maragua Town': 'Murang’a',
    'Kamahohu In Tetu': 'Nyeri',
    'Kasambara Village, Gilgil Sub': 'Nakuru',
    'Imenti North': 'Meru',
    'Kodumo East Sub-Location In Rachuonyo East Sub-County': 'Homa Bay',
    'Baba Ndogo Nairobi': 'Nairobi',
    'Plainsview In South B': 'Nairobi',
    'Nairovi': 'Nairobi',
    'Taveta': 'Taita Taveta',
    'Taita Tavet': 'Taita Taveta',
    'Njoro': 'Nakuru',
    "Murang'A South": 'Murang’a',
    'Marakwet': 'Elgeyo Marakwet',
    'Eldoret Townuasin Gishu': 'Uasin Gishu',
    'Uasin Gichu': 'Uasin Gishu',
    'N Airobi': 'Nairobi',
    'Nnadi': 'Nandi',
    'County': None,
    'Tran Nzoia': 'Trans Nzoia',
    'Nairobi.': 'Nairobi',
    'Nan': None}

# Final fix function
def final_fix(row):
    loc = row['Location'].split(',')[0].strip().title()
    return manual_fixes.get(row['Location'], town_to_county.get(loc, row['County']))

df_1_cleaned['County'] = df_1_cleaned.apply(final_fix, axis=1)

# Normalize known inconsistencies
county_fix_map = {
    'Muranga': 'Murang’a', 'Muranga County': 'Murang’a', "Murang'A": 'Murang’a',
    'Homabay': 'Homa Bay', 'Migory': 'Migori', 'Unknown': None, 'Nan': None, 'X': None}

df_1_cleaned['County'] = (
    df_1_cleaned['County']
    .replace(county_fix_map)
    .str.strip()
    .str.title())
```


```python
def final_county_cleanup(county_value):
    if pd.isna(county_value):
        return None
    cleaned = normalize_text(county_value)
    match, score, _ = process.extractOne(cleaned, normalized_counties, scorer=fuzz.ratio)
    if score > 85:
        return valid_counties[normalized_counties.index(match)]
    return None  
df_1_cleaned['County'] = df_1_cleaned['County'].apply(final_county_cleanup)

```


```python
df_1_cleaned['County'].unique()
```




    array(['Meru', 'Kakamega', 'Nairobi', 'Trans Nzoia', 'Kwale', 'Machakos',
           'Nakuru', None, 'Kitui', 'Kiambu', 'Kericho', 'Uasin Gishu',
           'Homa Bay', 'Murang’a', 'Nyamira', 'Laikipia', 'Bomet', 'Nyeri',
           'Kisumu', 'Taita Taveta', 'Siaya', 'Kisii', 'Bungoma', 'Mombasa',
           'Embu', 'Makueni', 'Busia', 'Isiolo', 'Nyandarua', 'Kajiado',
           'Kirinyaga', 'Elgeyo Marakwet', 'Mandera', 'Nandi', 'West Pokot',
           'Migori', 'Lamu', 'Baringo', 'Vihiga', 'Kilifi', 'Tana River',
           'Turkana', 'Garissa', 'Narok', 'Tharaka Nithi', 'Marsabit'],
          dtype=object)



#### SECOND DATASET


```python
import random
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)

# Define risk levels
risk_levels = ['safe', 'at_risk', 'immediate_danger']

# Sentences for each risk level
def generate_sentence(label, fake_case=False):
    emojis = {
        'safe': ['🙂', '💚', '✨', '🌻'],
        'at_risk': ['😟', '💔', '😥', '😓'],
        'immediate_danger': ['😰', '💢', '🆘', '💀']
    }
    children_mentions = [
        "my kids were home", "our daughter asked why we were yelling",
        "he never gets violent when the kids are around", "I told the children to stay in their room"
    ]
    calm_day = [
        "Today was peaceful", "We didn't argue at all today", "Everything felt calm for once", 
        "I even smiled today 🙂"
    ]
    at_risk_cases = [
        "He yelled at me again, but didn’t hit me", 
        "He came home drunk and punched the wall", 
        "He threw my phone across the room 💔", 
        "I locked myself in the bathroom to avoid him"
    ]
    danger_cases = [
        "He strangled me while I was screaming 🆘", 
        "I ran out barefoot after he pulled a knife 💀", 
        "He threatened to kill me and the kids", 
        "I sent my sister a voice note before he smashed my phone"
    ]
    manipulative_safe = [
        "I said he hit me but I just wanted him out", 
        "I reported him because I wanted custody of our child", 
        "I told my mom I was scared even though we were okay"
    ]
    
    base = ""
    if fake_case:
        base = random.choice(manipulative_safe)
    elif label == 'safe':
        base = random.choice(calm_day)
    elif label == 'at_risk':
        base = random.choice(at_risk_cases)
    elif label == 'immediate_danger':
        base = random.choice(danger_cases)
        
    # Occasionally add children context
    if random.random() < 0.3:
        base += ". " + random.choice(children_mentions)
        
    # Occasionally add emojis
    if random.random() < 0.6:
        base += " " + random.choice(emojis[label])
    
    return base.strip()

# Generate dataset
data = []
for label in risk_levels:
    for _ in range(500):
        is_fake = False
        if label in ['at_risk', 'immediate_danger'] and random.random() < 0.1:
            is_fake = True
        entry = {
            'text': generate_sentence(label, fake_case=is_fake),
            'risk_level': label,
            'age': random.randint(18, 65),
            'is_fake_report': is_fake
        }
        data.append(entry)

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle it
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Export to CSV
df.to_csv("femicide_simulated_dataset.csv", index=False)

print("Dataset generated and saved as 'femicide_simulated_dataset.csv'")

```

    Dataset generated and saved as 'femicide_simulated_dataset.csv'
    


```python

```

### DATA LOADING 


```python
#loading datset
df = pd.read_csv('femicide_simulated_dataset.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>risk_level</th>
      <th>age</th>
      <th>is_fake_report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I reported him because I wanted custody of our...</td>
      <td>immediate_danger</td>
      <td>57</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>He strangled me while I was screaming 🆘. our d...</td>
      <td>immediate_danger</td>
      <td>46</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I even smiled today 🙂 🙂</td>
      <td>safe</td>
      <td>20</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Today was peaceful. he never gets violent when...</td>
      <td>safe</td>
      <td>52</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Today was peaceful 🙂</td>
      <td>safe</td>
      <td>18</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (1500, 4)




```python
#checking for the class distribution
print(df['risk_level'].value_counts())
# Sample from each class
for label in df['risk_level'].unique():
    print(f"\n=== {label.upper()} ===")
    print(df[df['risk_level'] == label]['text'].sample(2).values)

```

    risk_level
    immediate_danger    500
    safe                500
    at_risk             500
    Name: count, dtype: int64
    
    === IMMEDIATE_DANGER ===
    ['He threatened to kill me and the kids'
     'He strangled me while I was screaming 🆘 💀']
    
    === SAFE ===
    ['Everything felt calm for once' 'I even smiled today 🙂 🌻']
    
    === AT_RISK ===
    ['He yelled at me again, but didn’t hit me. my kids were home 😟'
     'He threw my phone across the room 💔 💔']
    

### DATA CLEANING


```python

print("Missing values per column", df.isnull().sum())

# View percentage of missing values
missing_percent = df.isnull().mean() * 100
print("Missing % per column:", missing_percent)

# Display sample rows with missing values 
if df.isnull().values.any():
    display(df[df.isnull().any(axis=1)].head())
else:
    print("\n No missing values found.")
```

    Missing values per column text              0
    risk_level        0
    age               0
    is_fake_report    0
    dtype: int64
    Missing % per column: text              0.0
    risk_level        0.0
    age               0.0
    is_fake_report    0.0
    dtype: float64
    
     No missing values found.
    


```python
# delete the df_clean variable
if 'df_clean' in locals():
    del df_clean

# Reload the full original dataset 
df = pd.read_csv("femicide_simulated_dataset.csv")

# List the columns
print(df.columns)
df.head(10)

```

    Index(['text', 'risk_level', 'age', 'is_fake_report'], dtype='object')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>risk_level</th>
      <th>age</th>
      <th>is_fake_report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I reported him because I wanted custody of our...</td>
      <td>immediate_danger</td>
      <td>57</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>He strangled me while I was screaming 🆘. our d...</td>
      <td>immediate_danger</td>
      <td>46</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I even smiled today 🙂 🙂</td>
      <td>safe</td>
      <td>20</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Today was peaceful. he never gets violent when...</td>
      <td>safe</td>
      <td>52</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Today was peaceful 🙂</td>
      <td>safe</td>
      <td>18</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>He threw my phone across the room 💔 💔</td>
      <td>at_risk</td>
      <td>23</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I sent my sister a voice note before he smashe...</td>
      <td>immediate_danger</td>
      <td>63</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>He threw my phone across the room 💔. he never ...</td>
      <td>at_risk</td>
      <td>63</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>He strangled me while I was screaming 🆘</td>
      <td>immediate_danger</td>
      <td>48</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Everything felt calm for once 🙂</td>
      <td>safe</td>
      <td>51</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df['text'].head(10))  # Show first 10 rows
print(df['text'].apply(lambda x: isinstance(x, str)).value_counts())  # How many are valid strings?
print(df['text'].apply(lambda x: len(str(x).strip())).describe())  # Length of text values

```

    0    I reported him because I wanted custody of our...
    1    He strangled me while I was screaming 🆘. our d...
    2                              I even smiled today 🙂 🙂
    3    Today was peaceful. he never gets violent when...
    4                                 Today was peaceful 🙂
    5                He threw my phone across the room 💔 💔
    6    I sent my sister a voice note before he smashe...
    7    He threw my phone across the room 💔. he never ...
    8              He strangled me while I was screaming 🆘
    9                      Everything felt calm for once 🙂
    Name: text, dtype: object
    text
    True    1500
    Name: count, dtype: int64
    count    1500.000000
    mean       49.166667
    std        21.313041
    min        18.000000
    25%        37.000000
    50%        42.000000
    75%        61.000000
    max       106.000000
    Name: text, dtype: float64
    


```python
df.shape

```




    (1500, 4)




```python

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text
```


```python
df['clean_text'] = df['text'].apply(clean_text)

# See how many rows were reduced to empty
print("Empty after cleaning:", (df['clean_text'].str.strip() == '').sum())
print(df[['text', 'clean_text']].sample(5))  # Sample to verify visually

```

    Empty after cleaning: 0
                                                       text  \
    1327  He yelled at me again, but didn’t hit me. I to...   
    69                      Everything felt calm for once ✨   
    702   I even smiled today 🙂. I told the children to ...   
    189           He strangled me while I was screaming 🆘 🆘   
    176               He threatened to kill me and the kids   
    
                                         clean_text  
    1327   yelled didnt hit told children stay room  
    69                         everything felt calm  
    702   even smiled today told children stay room  
    189                         strangled screaming  
    176                        threatened kill kids  
    


```python
df['clean_text'].shape

```




    (1500,)



### EXPLORATORY DATA ANALYSIS

#####  Timeline of Femicide Cases Over Time


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Count cases by year
femicide_by_year = df_1_cleaned['date of murder'].dt.year.value_counts().sort_index()

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(x=femicide_by_year.index, y=femicide_by_year.values, marker='o')
plt.title("Femicide Cases in Kenya by Year of Murder", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Cases", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_44_0.png)
    


##### Monthly Distribution of Femicide Cases


```python
# Extract month names from 'date of murder'
df_1_cleaned['murder_month'] = df_1_cleaned['date of murder'].dt.month_name()

# Reorder months 
ordered_months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
monthly_counts = df_1_cleaned['murder_month'].value_counts().reindex(ordered_months)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette='Reds')
plt.title('Femicide Cases by Month')
plt.xlabel('Month')
plt.ylabel('Number of Cases')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_46_0.png)
    



```python
# Top 10 counties
top10_counties = df_1_cleaned['County'].value_counts().nlargest(10)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top10_counties.values, y=top10_counties.index, palette='Reds_r')

plt.title('Top 10 Counties with Most Femicide Cases')
plt.xlabel('Number of Cases')
plt.ylabel('County')
plt.tight_layout()
plt.show()
```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_47_0.png)
    



```python
import json
with open('kenya_constituencies.geojson','r',encoding='utf-8') as f:
    data = json.load(f)
print(data['features'][0]['properties'])
```

    {'OBJECTID': 1, 'COUNTY_NAM': 'NAIROBI', 'CONST_CODE': 288, 'CONSTITUEN': 'KAMUKUNJI', 'COUNTY_ASS': 0, 'COUNTY_A_1': '', 'REGIST_CEN': 0, 'REGISTRATI': '', 'COUNTY_COD': 47, 'Shape_Leng': 0.16463159484, 'Shape_Area': 0.00097789543, 'name': 'KAMUKUNJI'}
    


```python
# Build a mapping from county to list of geometries (polygons)
from collections import defaultdict

county_shapes = defaultdict(list)

for feature in data['features']:
    county = feature['properties']['COUNTY_NAM']
    geometry = feature['geometry']
    county_shapes[county].append(geometry)
```


```python
# Count number of cases per county in the data
county_counts = df_1_cleaned['County'].value_counts().reset_index()
county_counts.columns = ['County', 'Cases']
print(county_counts.head())
```

         County  Cases
    0   Nairobi     66
    1    Nakuru     59
    2    Kiambu     35
    3  Kakamega     17
    4  Murang’a     17
    


```python

correction_map = {
    "ELEGEYO MARAKWET": "ELGEYO MARAKWET",
    "THARAKA  NITHI": "THARAKA NITHI"}

# Function to normalize county names
def normalize_county_name(name):
    if not name:
        return ''
    name = name.strip().upper().replace('’', "'").replace("-", " ")
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with one
    return correction_map.get(name, name)  # Apply correction if exists

# Apply normalization to the DataFrame column
county_counts['County'] = county_counts['County'].apply(normalize_county_name)

# Check unmatched counties
geojson_counties = {normalize_county_name(f['properties']['COUNTY_NAM']) for f in data['features']}
dataframe_counties = set(county_counts['County'])

unmatched = geojson_counties - dataframe_counties
if unmatched:
    print("GeoJSON counties not in county_counts:", unmatched)
```

    GeoJSON counties not in county_counts: {'', 'SAMBURU', 'WAJIR'}
    


```python
# Group  data 
county_counts = df_1_cleaned['County'].value_counts().reset_index()
county_counts.columns = ['County', 'cases']
county_counts['County'] = county_counts['County'].apply(normalize_county_name)

```


```python
# Convert county_counts to dictionary for fast lookup
county_case_dict = dict(zip(county_counts['County'], county_counts['cases']))

combined = []
for feature in data['features']:
    county_name = normalize_county_name(feature['properties']['COUNTY_NAM'])
    case_count = county_case_dict.get(county_name, 0)
    combined.append({
        'county': county_name,
        'cases': case_count,
        'shapes': [feature['geometry']]
    })
print(combined[0])
```

    {'county': 'NAIROBI', 'cases': 66, 'shapes': [{'type': 'Polygon', 'coordinates': [[[36.87612497, -1.2838572], [36.87454353, -1.28124174], [36.87196269, -1.28362396], [36.86728922, -1.28753368], [36.86796498, -1.2891223], [36.86357252, -1.29116789], [36.86272068, -1.29130986], [36.86184407, -1.28913816], [36.85249232, -1.29332641], [36.85121747, -1.29447561], [36.84167725, -1.29240556], [36.83830859, -1.29205841], [36.83686856, -1.2899755], [36.8343485, -1.28920405], [36.82849836, -1.29213555], [36.82716118, -1.28953835], [36.83146843, -1.28740401], [36.83217559, -1.28589969], [36.83725428, -1.28566825], [36.83965863, -1.28494824], [36.84053293, -1.28442108], [36.84102152, -1.28379107], [36.84430017, -1.28305819], [36.84442874, -1.28043527], [36.84639116, -1.28030452], [36.84374289, -1.26765645], [36.84941599, -1.26633951], [36.85892076, -1.2660951], [36.85962682, -1.26481875], [36.86022427, -1.26446572], [36.86223384, -1.26370533], [36.86959324, -1.26150566], [36.87595373, -1.26253341], [36.87893507, -1.26245614], [36.88037436, -1.26821331], [36.88232962, -1.27712063], [36.87907085, -1.27809826], [36.88113474, -1.28222604], [36.87831047, -1.28203595], [36.87612497, -1.2838572]]]}]}
    


```python
from shapely.geometry import shape
import matplotlib.cm as cm
import matplotlib.colors as colors

fig, ax = plt.subplots(figsize=(12, 12))
max_cases = max(c['cases'] for c in combined)
norm = colors.Normalize(vmin=0, vmax=max_cases)
cmap = cm.Reds

for item in combined:
    for geom in item['shapes']:
        poly = shape(geom)
        color = cmap(norm(item['cases']))
        if poly.geom_type == 'Polygon':
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, edgecolor='black')
        elif poly.geom_type == 'MultiPolygon':
            for p in poly.geoms:
                x, y = p.exterior.xy
                ax.fill(x, y, color=color, edgecolor='black')


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Number of Femicide Cases')

plt.title('Femicide Cases per County', fontsize=16)
plt.axis('off')
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_54_0.png)
    


##### Age Distribution


```python
plt.figure(figsize=(8, 5))
sns.histplot(df_1_cleaned['Age'], bins=20, kde=True, color='salmon')
plt.title('Age Distribution of Femicide Victims')
plt.xlabel('Age')
plt

```




    <module 'matplotlib.pyplot' from 'C:\\Users\\HP\\anaconda3\\Lib\\site-packages\\matplotlib\\pyplot.py'>




    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_56_1.png)
    


##### Mode of Killing 


```python
plt.figure(figsize=(8, 5))
df_1_cleaned['Mode of killing'].value_counts().plot(kind='bar', color='crimson')
plt.title('Mode of Killing')
plt.xlabel('Method')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_58_0.png)
    


##### Suspect Relationship


```python
plt.figure(figsize=(8, 5))
df_1_cleaned['suspect relationship'].value_counts().plot(kind='bar', color='#8B0000')  # dark rich red
plt.title('Suspect Relationship to Victim')
plt.xlabel('Relationship')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_60_0.png)
    


##### Type of femicide


```python
plt.figure(figsize=(8, 5))
df_1_cleaned['Type of femicide'].value_counts().plot(kind='bar', color='#F08080')  # light coral
plt.title('Type of Femicide')
plt.xlabel('Femicide Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_62_0.png)
    



```python

df_1_cleaned['Circumstance'] = df_1_cleaned['Circumstance'].astype(str).str.strip()
filtered_df = df_1_cleaned[~df_1_cleaned['Circumstance'].str.lower().str.contains('unknown', na=False)]

# Create a Figure and Axes
fig, ax = plt.subplots(figsize=(8, 6))

# Plot with suspect relationship on X and circumstance as stacked categories
pd.crosstab(
    filtered_df['suspect relationship'],
    filtered_df['Circumstance']
).plot(
    kind='bar',
    stacked=True,
    colormap='Reds',
    ax=ax)

# Shift the plotting area within the figure
ax.set_position([0.1, 0.1, 0.6, 0.8])

# Move the legend out to the right
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

# Add labels
ax.set_title('Circumstance by Suspect Relationship (Excluding Unknown)')
ax.set_xlabel('Suspect Relationship')
ax.set_ylabel('Number of Cases')

plt.xticks(rotation=90, ha='right')
plt.tight_layout()
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_63_0.png)
    



```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of risk levels
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='risk_level', order=df['risk_level'].value_counts().index)
plt.title("Risk Level Distribution")
plt.ylabel("Count")
plt.xlabel("Risk Level")
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_64_0.png)
    



```python
# Fake report distribution
sns.countplot(data=df, x='is_fake_report')
plt.title("Fake vs Real Reports")
plt.show()

# Age distribution
df['age'].hist(bins=10, figsize=(6, 4))
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_65_0.png)
    



    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_65_1.png)
    


### FEATURE ENGINEERING


```python
#Risk key word counts
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
def count_words(text):
    return len(text.split())

def count_sentences(text):
    return text.count('.') + text.count('!') + text.count('?')

def count_all_caps(text):
    return sum(1 for word in text.split() if word.isupper())

def count_exclamations(text):
    return text.count('!')

# Risk keywords list (you can expand this)
risk_keywords = [
    'kill', 'hit', 'beat', 'threat', 'knife', 'gun', 'blood', 
    'abuse', 'danger', 'violence', 'escape', 'scream', 'stab', 'rape'
]

def contains_risk_keywords(text):
    text = text.lower()
    return any(keyword in text for keyword in risk_keywords)

df['clean_text'] = df['text'].apply(clean_text)
```


```python
y = df['risk_level']  
```


```python
#TF-IDF and Meta-features
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Basic Text Features
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['sentence_count'] = df['text'].apply(lambda x: len(re.split(r'[.!?]', str(x))) - 1)
df['uppercase_count'] = df['text'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()))
df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))

# Subjectivity & Polarity (Emotion)
df['subjectivity_score'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df['sentiment_score'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# 3. Risk Keyword Count
risk_keywords = ['kill', 'hurt', 'stab', 'beat', 'abuse', 'die', 'murder', 'choke', 'hit', 'rape', 'blood']
df['risk_keyword_count'] = df['text'].apply(
    lambda x: sum(1 for word in str(x).lower().split() if word in risk_keywords)
)

# Sentiment vs Risk Level Mismatch
def check_mismatch(row):
    if row['risk_level'] == 'safe' and row['sentiment_score'] < -0.2:
        return 1
    elif row['risk_level'] == 'immediate_danger' and row['sentiment_score'] > 0.2:
        return 1
    else:
        return 0

df['sentiment_risk_mismatch'] = df.apply(check_mismatch, axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
X_tfidf = vectorizer.fit_transform(df['text'])
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Combine All Features
meta_features = df[[
    'word_count', 'sentence_count', 'uppercase_count', 'exclamation_count',
    'subjectivity_score', 'sentiment_score', 'risk_keyword_count',
    'sentiment_risk_mismatch'
]].reset_index(drop=True)

X_final = pd.concat([meta_features, tfidf_df], axis=1)

# Target
y = df['risk_level']

# Print shape 
print("Final feature shape:", X_final.shape)
print("Target distribution:\n", y.value_counts())

```

    Final feature shape: (1500, 157)
    Target distribution:
     risk_level
    immediate_danger    500
    safe                500
    at_risk             500
    Name: count, dtype: int64
    


```python
#Sentiment score
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='risk_level', y='sentiment_score', palette='Set2')
plt.title('Sentiment Score Distribution by Risk Level')
plt.xlabel('Risk Level')
plt.ylabel('Sentiment Score (TextBlob)')
plt.grid(True)
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_70_0.png)
    



```python
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='risk_level', y='risk_keyword_count', palette='Set3')
plt.title('Risk Keyword Count by Risk Level')
plt.xlabel('Risk Level')
plt.ylabel('Number of Risk Keywords')
plt.grid(True)
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_71_0.png)
    



```python
meta_features = ['sentiment_score', 'risk_keyword_count', 'word_count', 'sentence_count']

plt.figure(figsize=(8, 6))
sns.heatmap(df[meta_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Meta Features')
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_72_0.png)
    



```python
# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the features
X_scaled = StandardScaler().fit_transform(df[meta_features])

# PCA to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

# Plot PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='risk_level', palette='Set1')
plt.title('PCA Visualization of Meta Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

```


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_73_0.png)
    



```python
df['clean_text'].shape

```




    (1500,)



### MODELING 

#### Logistic Regression


```python

# Features & target
X = df['clean_text']   # keep emojis
y = df['risk_level']   # safe, at_risk, immediate_danger

# Train-test split (stratify to keep balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# Define the simplified pipeline
pipeline_model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Define the hyperparameters to tune. 
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)], # Test unigrams and bigrams
    'clf__C': [0.1, 1.0, 10.0], # Regularization parameter
}

# GridSearchCV to find the best parameters
grid = GridSearchCV(pipeline_model, param_grid, cv=3, verbose=1, n_jobs=-1, scoring='f1_macro')
grid.fit(X_train, y_train)

# The rest of the notebook can use grid_search.best_estimator_ instead of pipeline_model
best_model = grid.best_estimator_


```

    Fitting 3 folds for each of 6 candidates, totalling 18 fits
    

#### Random Forest


```python
# Split
X = df[['clean_text', 'word_count', 'sentiment_score']]  # Example features
y = df['risk_level']
X_train_rm, X_test_rm, y_train_rm, y_test_rm = train_test_split(X, y, stratify=y, random_state=42)

# Define vectorizer and preprocessor
text_features = 'clean_text'
meta_features = ['word_count',  'sentiment_score']

preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(), text_features),
    ('meta', StandardScaler(), meta_features)
])

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Fit
pipeline.fit(X_train_rm, y_train_rm)

```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;text&#x27;, TfidfVectorizer(),
                                                  &#x27;clean_text&#x27;),
                                                 (&#x27;meta&#x27;, StandardScaler(),
                                                  [&#x27;word_count&#x27;,
                                                   &#x27;sentiment_score&#x27;])])),
                (&#x27;classifier&#x27;,
                 RandomForestClassifier(class_weight=&#x27;balanced&#x27;,
                                        random_state=42))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('steps',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">steps&nbsp;</td>
            <td class="value">[(&#x27;preprocessor&#x27;, ...), (&#x27;classifier&#x27;, ...)]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transform_input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">transform_input&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('memory',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">memory&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>preprocessor: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformers',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">transformers&nbsp;</td>
            <td class="value">[(&#x27;text&#x27;, ...), (&#x27;meta&#x27;, ...)]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('remainder',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">remainder&nbsp;</td>
            <td class="value">&#x27;drop&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('sparse_threshold',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">sparse_threshold&nbsp;</td>
            <td class="value">0.3</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_jobs&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformer_weights',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">transformer_weights&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose_feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose_feature_names_out&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('force_int_remainder_cols',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">force_int_remainder_cols&nbsp;</td>
            <td class="value">&#x27;deprecated&#x27;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>text</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__text__"><pre>clean_text</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>TfidfVectorizer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html">?<span>Documentation for TfidfVectorizer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__text__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">input&nbsp;</td>
            <td class="value">&#x27;content&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoding',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">encoding&nbsp;</td>
            <td class="value">&#x27;utf-8&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('decode_error',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">decode_error&nbsp;</td>
            <td class="value">&#x27;strict&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strip_accents',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">strip_accents&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('lowercase',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">lowercase&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('preprocessor',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">preprocessor&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tokenizer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">tokenizer&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('analyzer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">analyzer&nbsp;</td>
            <td class="value">&#x27;word&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('stop_words',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">stop_words&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('token_pattern',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">token_pattern&nbsp;</td>
            <td class="value">&#x27;(?u)\\b\\w\\w+\\b&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ngram_range',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">ngram_range&nbsp;</td>
            <td class="value">(1, ...)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_df&nbsp;</td>
            <td class="value">1.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_df&nbsp;</td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_features&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('vocabulary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">vocabulary&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('binary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">binary&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">dtype&nbsp;</td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('norm',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">norm&nbsp;</td>
            <td class="value">&#x27;l2&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('use_idf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">use_idf&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('smooth_idf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">smooth_idf&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('sublinear_tf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">sublinear_tf&nbsp;</td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>meta</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__meta__"><pre>[&#x27;word_count&#x27;, &#x27;sentiment_score&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__meta__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">copy&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">with_mean&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">with_std&nbsp;</td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="classifier__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_estimators',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_estimators&nbsp;</td>
            <td class="value">100</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('criterion',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">criterion&nbsp;</td>
            <td class="value">&#x27;gini&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_depth',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_depth&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_split',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_samples_split&nbsp;</td>
            <td class="value">2</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_samples_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_samples_leaf&nbsp;</td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_weight_fraction_leaf',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_weight_fraction_leaf&nbsp;</td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_features&nbsp;</td>
            <td class="value">&#x27;sqrt&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_leaf_nodes',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_leaf_nodes&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_impurity_decrease',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_impurity_decrease&nbsp;</td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('bootstrap',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">bootstrap&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('oob_score',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">oob_score&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_jobs&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">random_state&nbsp;</td>
            <td class="value">42</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">warm_start&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('class_weight',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">class_weight&nbsp;</td>
            <td class="value">&#x27;balanced&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ccp_alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">ccp_alpha&nbsp;</td>
            <td class="value">0.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_samples',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_samples&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('monotonic_cst',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">monotonic_cst&nbsp;</td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>



### EVALUATION


```python
# Evaluation of Logistic Regression

# Evaluation
y_pred = grid.predict(X_test)

print("Classification Report:", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred, labels=grid.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=grid.classes_,
            yticklabels=grid.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
```

    Classification Report:                   precision    recall  f1-score   support
    
             at_risk       1.00      0.91      0.95       100
    immediate_danger       0.92      1.00      0.96       100
                safe       1.00      1.00      1.00       100
    
            accuracy                           0.97       300
           macro avg       0.97      0.97      0.97       300
        weighted avg       0.97      0.97      0.97       300
    
    


    
![png](Femicide_Prevention%20%286%29_files/Femicide_Prevention%20%286%29_81_1.png)
    



```python
# Evaluation of random forest
from sklearn.metrics import classification_report
y_pred_rm = pipeline.predict(X_test_rm)
print(classification_report(y_test_rm, y_pred_rm))

```

                      precision    recall  f1-score   support
    
             at_risk       0.97      0.94      0.95       125
    immediate_danger       0.94      0.97      0.95       125
                safe       1.00      1.00      1.00       125
    
            accuracy                           0.97       375
           macro avg       0.97      0.97      0.97       375
        weighted avg       0.97      0.97      0.97       375
    
    

### DistilBERT


```python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# Prepare data
df = df[['clean_text', 'risk_level']].copy()
label_map = {label: idx for idx, label in enumerate(sorted(df["risk_level"].unique()))}
df["label"] = df["risk_level"].map(label_map)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["clean_text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["clean_text", "risk_level"])
test_dataset = test_dataset.remove_columns(["clean_text", "risk_level"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=len(label_map)
)

# FIX: remove unsupported keyword
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",   # old: evaluation_strategy
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True)

def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics)

trainer.train()

#Evaluate model
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)



```


    Map:   0%|          | 0/1200 [00:00<?, ? examples/s]



    Map:   0%|          | 0/300 [00:00<?, ? examples/s]


    Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    



    <div>

      <progress value='450' max='450' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [450/450 35:16, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.287000</td>
      <td>0.109103</td>
      <td>0.966667</td>
      <td>0.966586</td>
      <td>0.967697</td>
      <td>0.966667</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.060500</td>
      <td>0.100879</td>
      <td>0.966667</td>
      <td>0.966586</td>
      <td>0.967697</td>
      <td>0.966667</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.063500</td>
      <td>0.108445</td>
      <td>0.966667</td>
      <td>0.966586</td>
      <td>0.967697</td>
      <td>0.966667</td>
    </tr>
  </tbody>
</table><p>




<div>

  <progress value='38' max='38' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [38/38 00:42]
</div>



    Evaluation metrics: {'eval_loss': 0.10087945312261581, 'eval_accuracy': 0.9666666666666667, 'eval_f1': 0.966585638144627, 'eval_precision': 0.9676968774904007, 'eval_recall': 0.9666666666666667, 'eval_runtime': 43.793, 'eval_samples_per_second': 6.85, 'eval_steps_per_second': 0.868, 'epoch': 3.0}
    

#### Model Interpretation with SHAP


```python

import shap
import numpy as np
import torch
# Get a sample text from the test set
X_test_text = test_df['clean_text'].iloc[[0]]  # double brackets keep proper shape

# Define a function that the SHAP explainer can use to get predictions from the model
def predict_proba_from_text(texts):
    # Tokenize the input text
    encoded = tokenizer(
        list(texts),  # works for Series, list, or numpy array
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    # Move to same device as the model
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Get predictions from the model
    with torch.no_grad():
        logits = model(**encoded).logits

    # Apply softmax to get probabilities
    return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

# Create a SHAP text masker (important for correct token handling)
masker = shap.maskers.Text(tokenizer)

# Create a SHAP explainer with the masker
explainer = shap.Explainer(predict_proba_from_text, masker, algorithm="partition")

# Calculate SHAP values for the sample text
shap_values = explainer(X_test_text)

# The labels from your `label_map`
class_names = list(label_map.keys())

```

    PartitionExplainer explainer: 2it [00:14, 14.10s/it]           
    


```python
# Save the model 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model.save_pretrained("./distilbert_model_v2", safe_serialization=False)
tokenizer.save_pretrained("./distilbert_model_v2")
```




    ('./distilbert_model_v2\\tokenizer_config.json',
     './distilbert_model_v2\\special_tokens_map.json',
     './distilbert_model_v2\\vocab.txt',
     './distilbert_model_v2\\added_tokens.json',
     './distilbert_model_v2\\tokenizer.json')




```python

```


```python

```


```python

```
