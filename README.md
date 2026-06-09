# Differential Privacy and Data Anonymization: A Red Team / Blue Team Study

## Project Overview

The goal of this project is to study the tradeoff between privacy and utility when releasing sensitive datasets. Organizations often wish to share data with researchers or the public, but doing so creates the risk that individuals may be identified or that sensitive information may be inferred. While techniques such as anonymization and differential privacy are intended to reduce these risks, they can also reduce the usefulness of the released data.

Rather than focusing solely on predictive performance, this project will evaluate privacy mechanisms from both a defender's and an attacker's perspective. The project is inspired by the common red team / blue team framework used in cybersecurity.

The blue team is responsible for releasing a useful version of a dataset while protecting the privacy of individuals. The red team is responsible for attempting to recover sensitive information or identify individuals using the released data.

The central question is:

**How much utility can be preserved while preventing realistic privacy attacks?**

## Potential Datasets

Several datasets are being considered.

### Adult Income Dataset

The Adult Income dataset from the UCI Machine Learning Repository contains demographic and socioeconomic information such as age, education, occupation, marital status, race, sex, and income category.

**Advantages:**
- Widely used benchmark dataset.
- Contains sensitive demographic information.
- Supports machine learning experiments.
- Suitable for re-identification and membership inference studies.

### Diabetes 130-US Hospitals Dataset

This dataset contains patient-level hospital records and readmission outcomes.

**Advantages:**
- Strong privacy motivation.
- Healthcare data is a major application area for privacy-preserving analysis.
- Membership inference attacks have clear real-world implications.

### Credit Card Default Dataset

This dataset contains customer financial information and loan default outcomes.

**Advantages:**
- Sensitive financial attributes.
- Supports classification tasks.
- Natural setting for studying privacy risks in financial data.

At present, the Adult Income dataset appears to be the strongest candidate because it supports all proposed experiments while remaining manageable in size.

## Stakeholders

The project is relevant to several groups:

- Organizations releasing data for research purposes.
- Researchers who rely on access to high-quality datasets.
- Individuals whose records appear in released datasets.
- Policymakers and regulators concerned with privacy protection.

## Unit of Analysis

The unit of analysis will generally be an individual record.

Examples include:
- One individual in the Adult Income dataset.
- One patient encounter in the Diabetes dataset.
- One customer in the Credit Card dataset.

Differential privacy is naturally defined at the level of an individual record, making these datasets suitable for analysis.

## Blue Team Objectives

The blue team's goal is to release a useful version of the data while reducing privacy risks.

Potential privacy-preserving approaches include:
- Removal of direct identifiers.
- Generalization of attributes.
- Suppression of rare values.
- k-anonymity.
- l-diversity.
- Differential privacy.
- Differentially private synthetic data generation.

The blue team will attempt to maximize utility while maintaining acceptable privacy guarantees.

## Red Team Objectives

The red team's goal is to determine how much information can be recovered from the released dataset.

Possible attacks include:

### Re-identification Attacks

Attempt to identify individuals using combinations of quasi-identifiers such as age, sex, occupation, or geographic information.

### Membership Inference Attacks

Attempt to determine whether a specific individual was included in a training dataset.

### Attribute Inference Attacks

Attempt to infer sensitive attributes that were not directly released.

The effectiveness of these attacks will be used as a measure of privacy risk.

## Project Directions

Several experimental directions are possible.

### Option 1: Differentially Private Machine Learning

Train machine learning models with varying privacy budgets and compare their performance against standard models.

**Potential models:**
- Logistic Regression
- Random Forest
- Differentially Private Logistic Regression
- Differentially Private SGD

**Metrics:**
- Accuracy
- F1 Score
- Privacy budget (epsilon)

### Option 2: Differentially Private Synthetic Data

Generate synthetic datasets using differential privacy techniques and evaluate how closely they resemble the original dataset.

**Metrics:**
- Distribution preservation
- Correlation preservation
- Downstream model performance
- Privacy budget (epsilon)

### Option 3: Red Team / Blue Team Evaluation

Compare multiple privacy mechanisms under active attack.

**Potential release methods:**
- Original dataset
- Basic anonymization
- k-anonymity
- Differential privacy
- Differentially private synthetic data

**Potential attack metrics:**
- Re-identification success rate
- Membership inference success rate
- Attribute inference success rate

This option provides the most direct evaluation of privacy protection.

## Data Assessment

The candidate datasets contain sufficient observations for machine learning and privacy experiments.

- Adult Income contains approximately 49,000 observations and 14 features.
- The Credit Card Default dataset contains approximately 30,000 observations.
- The Diabetes dataset contains over 100,000 records.

Potential sources of bias and representativeness issues will be examined during exploratory analysis.

## Learnability Assessment

Before implementing privacy-preserving methods, baseline models will be trained to establish whether meaningful predictive signal exists in the data.

**Baseline models include:**
- DummyClassifier
- Logistic Regression
- Random Forest

Cross-validation will be used to evaluate performance.

If predictive models substantially outperform trivial baselines, the prediction task will be considered learnable.

## Privacy Threat Model

The project assumes an adversary with access to released datasets and potentially some auxiliary information.

The adversary may attempt to:
- Determine whether an individual participated in the dataset.
- Re-identify specific individuals.
- Infer sensitive information from released records.

The effectiveness of privacy-preserving mechanisms will be evaluated against these attack models.

## Key Performance Indicators

### Utility Metrics
- Accuracy
- F1 Score
- Statistical estimation error
- Distribution similarity
- Correlation preservation

### Privacy Metrics
- Privacy budget (epsilon)
- Re-identification rate
- Membership inference attack success rate
- Attribute inference attack success rate

### Computational Metrics
- Runtime
- Memory usage
- Training cost

## Expected Deliverables

- README describing the problem and datasets.
- Data acquisition and preprocessing scripts.
- Baseline modeling notebook.
- Privacy-preserving data release experiments.
- Red team attack implementations.
- Results tables and visualizations.
- Final report discussing privacy-utility tradeoffs and attack outcomes.

At the conclusion of the project, we hope to identify which privacy-preserving techniques provide the best balance between data utility and resistance to realistic attacks.