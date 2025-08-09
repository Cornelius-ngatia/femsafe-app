# Predictive Femicide Risk Detection in Kenya

## Problem Statement
Femicide ,the gender-based killing of women , is a growing concern in Kenya, with over 150 cases reported in 2023 alone.  
These deaths are often the culmination of repeated gender-based violence (GBV), emotional abuse, and systemic failures to intervene.  
Traditional response mechanisms are reactive, often triggered too late.

**Goal:**  
To create a **data-driven, preventative system** that flags high-risk cases early and guides timely interventions from law enforcement, social workers, NGOs, and communities.


## Stakeholders

- **NGOs & Civil Society Organizations**   
  Provide shelters, support survivors, and use risk assessment tools for early interventions.
- **Law Enforcement Agencies**  
  Identify high-risk individuals or regions and prioritize protective action.
- **Healthcare Providers**  
  Flag repeated injuries, mental health issues, and delayed reporting patterns.
- **Judiciary & Legal Practitioners**  
  Integrate risk indicators into court decisions and protection orders.
- **Government & Policy Makers**  
  Allocate resources effectively and design GBV prevention policies.
- **Survivors & Communities**  
  Receive better protection through proactive risk detection.


## Objectives
1. **AI-Powered Risk Detection** Analyze reported incidents and behavioral patterns to detect early signs of GBV/femicide risk.  
2. **Safety Features** Real-time alerts, panic button, and connection to shelters, counseling, and legal aid.  
3. **Awareness & Education** Inform users on recognizing abuse, knowing their rights, and building safety plans.

## Datasets
- **First Dataset** — Historical femicide cases in Kenya with temporal, geographic, and demographic details.  
- **Second Dataset** — Incident narratives and additional socio-economic factors for NLP modeling. 

## Project Workflow

### Data Loading
Import datasets and check for structure, data types, and completeness.

### Data Cleaning
- Handle missing values  
- Normalize categorical labels  
- Remove duplicates  
- Standardize date formats

### Exploratory Data Analysis (EDA)
Visual insights generated for:
- **Timeline of Femicide Cases Over Time**
- **Monthly Distribution**
- **Age Distribution**
- **Mode of Killing**
- **Suspect Relationship**
- **Type of Femicide**

### Feature Engineering
- Encoding categorical variables  
- Creating derived features from dates  
- Text vectorization for narrative data

###  Modeling
Models implemented:
- **Logistic Regression**
- **Random Forest**
- **DistilBERT** 
### Model Evaluation
Metrics used:
- Accuracy  
- Precision  
- Recall  
- F1-score  
SHAP was used for **model interpretability**.

## Results
| Model             | Accuracy | Precision | Recall | F1-score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 97%      | 100%       | 91%    | 95%      |
| Random Forest     | 97%      | 97%       | 94%    | 95%      |
| DistilBERT        | 97%      | 97%       | 97%    |   97%  |


## Ethical Considerations
- **Data Privacy**  Ensure sensitive victim data is anonymized.
- **Bias Mitigation** Regular bias audits of ML models.
- **Responsible Use** Predictions are for **support**, not punitive measures.

## Deployment
https://femsafe-app-dgsfqrxeycprh3gta8sufn.streamlit.app/
