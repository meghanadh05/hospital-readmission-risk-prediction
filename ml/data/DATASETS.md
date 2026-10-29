# Datasets used in this project

| Dataset | Where | Rows | Role |
|---|---|---|---|
| Diabetes 130-US hospitals (`diabetic_data.csv`, `description.pdf`) | ml/data/ (included) | 101,766 encounters -> 70,434 after cleaning | **Main training data** - has the 30-day readmission label |
| AV Healthcare Analytics II (`av_healthcare/`) | ml/data/av_healthcare/ (included) | 318,438 train / 137,057 test | **Supporting** - predicts length of stay, not readmission |
| MIMIC-III v1.4 | NOT included (licence) | ~58k admissions | **Optional extension** - load with `python ml/train.py <mimic_folder>` |

## 1. Diabetes 130-US hospitals (Kaggle, brandao/diabetes)
Records of diabetic patients from 130 US hospitals over 10 years (1999-2008), 50 columns: demographics, admission/discharge type,
days in hospital, lab/procedure/medication counts, prior-year outpatient/emergency/inpatient visits, 3 ICD-9 diagnoses, A1C and
glucose results, 23 drug columns, and `readmitted` (`<30`, `>30`, `NO`).
Preprocessing (from the reference notebook): drop expired patients and unknown gender, keep each patient's first encounter,
group ICD-9 into 9 disease groups, add `service_utilization` and `numchange`. Target = readmitted `<30` days (8.9% of encounters).

## 2. AV Healthcare Analytics II (Kaggle)
Hospital-stay records with: hospital code/type/city/region, available extra rooms, department, ward type, bed grade, patient id,
admission type, severity of illness, visitors, age band, admission deposit, and **Stay** (length-of-stay band: 0-10 ... 91-100, More than 100 Days).
It has no readmission label, so it is not used for training the readmission model. Possible use: a length-of-stay
predictor to help bed planning (a longer expected stay is another risk signal).

## 3. MIMIC-III v1.4 (PhysioNet) - https://physionet.org/content/mimiciii/1.4/
A large, freely available but access-controlled critical-care database from Beth Israel Deaconess Medical Center (Boston), 2001-2012:
about 46,500 patients and 58,000 hospital admissions (incl. 61,000 ICU stays), de-identified (dates shifted, ages over 89 masked).
Tables include ADMISSIONS, PATIENTS, DIAGNOSES_ICD, PROCEDURES_ICD, PRESCRIPTIONS, LABEVENTS, CHARTEVENTS, NOTEEVENTS.
Access: create a PhysioNet account, complete the CITI "Data or Specimens Only Research" training, sign the data use agreement,
and apply for credentialing. It cannot be redistributed, so it is not in this zip.
`ml/mimic.py` builds the same feature table from ADMISSIONS + PATIENTS + DIAGNOSES_ICD (+ PROCEDURES_ICD, PRESCRIPTIONS if present),
labelling an admission as readmitted when the same patient is admitted again within 30 days of discharge.
