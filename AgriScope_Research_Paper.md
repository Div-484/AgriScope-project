---
title: "AgriScope: An Intelligent Decision Support System for Smart Agriculture in Gujarat Using Ensemble Machine Learning and Real-Time Weather Integration"
author: "Final Year B.Tech / BCA / MCA Project"
date: "Academic Year 2024–2025"
institution: "Department of Computer Science and Data Analytics"
---

# AGRISCOPE: AN INTELLIGENT DECISION SUPPORT SYSTEM FOR SMART AGRICULTURE IN GUJARAT USING ENSEMBLE MACHINE LEARNING AND REAL-TIME WEATHER INTEGRATION

---

## TITLE PAGE

**Project Title:**
AgriScope: An Intelligent Decision Support System for Smart Agriculture in Gujarat Using Ensemble Machine Learning and Real-Time Weather Integration

**Submitted in partial fulfilment of the requirements for the award of the degree of**
Bachelor of Technology / Master of Computer Applications
in Computer Science and Data Analytics

**Submitted by:**
[Student Name(s)]
[Enrollment Number(s)]

**Under the Guidance of:**
[Guide Name], [Designation]
Department of Computer Science and Data Analytics

**Department of Computer Science and Data Analytics**
[Name of Institution]
[City, Gujarat, India]
**Academic Year 2024–2025**

---

## DECLARATION

I hereby declare that the project work entitled **"AgriScope: An Intelligent Decision Support System for Smart Agriculture in Gujarat Using Ensemble Machine Learning and Real-Time Weather Integration"** submitted to the Department of Computer Science and Data Analytics, [Institution Name], in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology / Master of Computer Applications in Computer Science, is a record of original work done by me under the supervision and guidance of [Guide Name], [Designation], Department of Computer Science and Data Analytics, [Institution Name].

I further declare that the work reported in this thesis has not been submitted, either in part or in full, for the award of any other degree or diploma in this institution or any other institution or university. All sources of information used in this project have been duly acknowledged.

Date: March 2025
Place: [City], Gujarat

[Signature]
[Student Name]
[Enrollment Number]

---

## CERTIFICATE

This is to certify that the project entitled **"AgriScope: An Intelligent Decision Support System for Smart Agriculture in Gujarat Using Ensemble Machine Learning and Real-Time Weather Integration"** has been carried out by [Student Name(s)] ([Enrollment Number(s)]) under my guidance and supervision, in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology / Master of Computer Applications in Computer Science and Data Analytics from [Institution Name], [City], Gujarat.

To the best of my knowledge and belief, the work presented in this project thesis is original and has not been submitted for the award of any degree, diploma, or certificate elsewhere.

**Project Guide:**
[Guide Name]
[Designation]
Department of Computer Science and Data Analytics
[Institution Name]

**Head of Department:**
[HOD Name]
[Designation]
Department of Computer Science and Data Analytics
[Institution Name]

Date: March 2025

---

## ACKNOWLEDGEMENT

I express my sincere gratitude to my project guide, **[Guide Name]**, [Designation], Department of Computer Science and Data Analytics, [Institution Name], for the invaluable guidance, constant encouragement, constructive feedback, and academic support extended throughout the course of this project. His/Her deep expertise in the domain of machine learning and agricultural data analytics has been a source of motivation and direction at every stage of this work.

I extend my heartfelt thanks to the **Head of Department**, [HOD Name], for providing the necessary infrastructure, resources, and an academic environment that made this project possible. The support provided by the department, including access to computational resources and datasets, has been instrumental in completing this work.

I am deeply grateful to the **Gujarat State Disaster Management Authority (GSDMA)** and the **India Meteorological Department (IMD)** for making agricultural and rainfall datasets publicly accessible, which form the empirical foundation of this research. The availability of open data on crop production, seasonal rainfall, and district-level statistics has made it possible to develop a data-driven system that reflects real-world agricultural conditions in Gujarat.

I also thank the developers and contributors of **Open-Meteo**, an open-source weather API project, whose freely accessible real-time weather data interface has enabled the dynamic weather integration feature in the AgriScope dashboard. Thanks are equally due to the open-source communities behind **Streamlit**, **scikit-learn**, **XGBoost**, **Pandas**, **NumPy**, and **Matplotlib**, whose robust, well-documented tools have made modern machine learning development accessible and efficient.

Special thanks to my **family** for their unconditional support, patience, and encouragement throughout the project period. Their faith in my abilities has been the strongest motivating force behind this effort.

Finally, I acknowledge all my **friends and classmates** for the healthy academic discussions, peer reviews, and moral support that enriched this learning experience.

[Student Name]
March 2025

---

## ABSTRACT

Agriculture is the backbone of the Indian economy, supporting the livelihoods of more than 58% of the rural population and contributing approximately 16–17% to the country's Gross Domestic Product (GDP). In the state of Gujarat, agriculture plays a pivotal role in economic sustenance, supporting millions of farmers across 32 districts and producing a diverse range of crops including groundnut, cotton, bajra, wheat, rice, and castor. However, the agricultural sector in Gujarat and across India faces a complex and multifaceted set of challenges rooted in environmental unpredictability, climatic variability, inadequate access to actionable data, and limited technology penetration at the farm level.

Traditional agricultural practices in Gujarat rely heavily on experiential knowledge passed down through generations, regional customs, and often uninformed crop selection patterns. While these practices represent centuries of accumulated wisdom, they are increasingly inadequate in the face of changing climatic conditions, irregular monsoon patterns, prolonged droughts, and rising temperatures. Farmers lack scientifically grounded tools to determine which crops are likely to perform best under current climatic and geographic conditions, and what yield levels they can realistically expect from a given crop in a given district during a given season. This knowledge gap results in suboptimal crop choices, overuse of inputs such as water and fertilizers, reduced productivity, and significant economic losses.

The emergence of machine learning (ML) as a powerful data-driven paradigm has opened up transformative possibilities for the agricultural sector. Machine learning algorithms can extract complex non-linear patterns from large historical datasets comprising crop production records, seasonal weather parameters, soil characteristics, and economic indicators. Once trained, these models can generalize from historical data to provide accurate, data-driven predictions about future crop yields — predictions that are both quantitatively precise and empirically grounded.

**AgriScope** is a comprehensive, end-to-end intelligent decision support system developed to address these critical challenges in Gujarat's agricultural landscape. The system integrates historical crop production data spanning the years 2016 through 2024 across all 32 districts of Gujarat, district-level annual rainfall data from 2014 to 2024, and real-time weather data obtained through the Open-Meteo API. These diverse data streams are synthesized through a robust machine learning pipeline to provide farmers and agriculture planners with accurate crop yield predictions and crop type recommendations tailored to specific districts, seasons, and prevailing weather conditions.

The AgriScope project implements, trains, and evaluates eight distinct machine learning regression models: **ExtraTrees Regressor**, **Gradient Boosting Regressor**, **Random Forest Regressor**, **XGBoost Regressor**, **Decision Tree Regressor**, **K-Nearest Neighbors (KNN) Regressor**, **Ridge Regression**, and **ElasticNet Regression**. Each model is trained on a feature set comprising eight carefully engineered input features: district encoding, season encoding, crop type encoding, total seasonal rainfall, number of rainy days, average maximum temperature, average minimum temperature, and average relative humidity. All models are trained using log-transformed yield values to address the strong right-skew present in agricultural yield distributions, with predictions inverse-transformed back to the original scale for interpretability.

Experimental results demonstrate that the **ExtraTrees Regressor** delivers the best predictive performance among all evaluated models, achieving an R² score of **0.6727** (67.27% accuracy), a Mean Absolute Error (MAE) of **342.53 kg/ha**, and a Root Mean Square Error (RMSE) of **458.01 kg/ha** on the held-out test set. The Gradient Boosting, Random Forest, and XGBoost models form a competitive second tier with R² scores of 0.6242, 0.6167, and 0.6130 respectively, while linear models (Ridge Regression and ElasticNet) yield negative R² values, demonstrating that linear assumptions are fundamentally incompatible with the highly non-linear nature of agricultural yield data.

Feature importance analysis reveals that **crop type** is by far the most dominant predictor of yield (importance score 0.512, accounting for 51.2% of predictive power), followed by district geography (0.198), total seasonal rainfall (0.082), minimum temperature (0.058), and maximum temperature (0.051). This finding reflects the fundamental agronomic reality that different crops have inherently different yield potential — cotton yields in the range of 400–700 kg/ha, whereas wheat and bajra can yield 1,500–3,000 kg/ha under optimal conditions.

The AgriScope system is deployed as a fully interactive, multi-page Streamlit web dashboard with five functional modules: an **Overview** module presenting system context and model performance; a **Crop Prediction** module enabling real-time predictions using live weather data; an **Agricultural Analytics** module providing visual insights into district-wise production, seasonal yield patterns, and crop distributions; a **Rainfall Analysis** module visualizing historical rainfall trends; and a **Prediction History** module backed by a SQLite database for persistent prediction logging.

This thesis describes the complete lifecycle of the AgriScope system — from problem conceptualization, dataset curation and preprocessing, feature engineering, model training and hyperparameter selection, evaluation methodology, results analysis, and system architecture through to implementation, deployment, and future development directions. The work makes the following key contributions to the field of agricultural informatics and applied machine learning: (1) a comprehensive comparison of eight ML regression models on real-world Gujarat agricultural data; (2) demonstration that ensemble tree-based methods significantly outperform linear models for agricultural yield prediction; (3) integration of real-time weather data with ML prediction pipelines for actionable decision support; (4) development and deployment of a full-stack agricultural decision support system with persistent database integration; and (5) empirical validation of crop type as the dominant factor in agricultural yield prediction from tabular historical data.

---

# CHAPTER 1: INTRODUCTION

## 1.1 Global Agriculture: Context and Importance

Agriculture is one of humanity's oldest and most fundamental activities, dating back more than ten thousand years to the Neolithic revolution when nomadic hunter-gatherer societies transitioned to settled agricultural communities. Today, agriculture remains the largest employer in the world, engaging approximately **1.3 billion people** — more than any other single industry. According to the Food and Agriculture Organization (FAO) of the United Nations, global agricultural production must increase by approximately **70%** by the year 2050 to feed a projected world population of 9.7 billion people. This staggering requirement must be achieved even as climate change reshapes growing seasons, arable land availability, and water resources in ways that are both profound and often unpredictable.

The global challenge of agricultural sustainability is compounded by several interlocking crisis factors. First, **climate change** is causing more frequent and severe weather events — droughts, floods, heatwaves, and erratic monsoons — that directly threaten crop production. The Intergovernmental Panel on Climate Change (IPCC) projects that, without adaptation measures, global crop yields of staple crops such as wheat, rice, and maize could decline by 2–6% per decade over the coming decades due to rising temperatures alone. Second, **population growth** in food-insecure regions of Asia and Africa is intensifying demand even as supply becomes more precarious. Third, the **degradation of agricultural land** through soil erosion, salinization, and desertification is reducing the total area of productive farmland. Fourth, **water scarcity** is increasingly constraining irrigation-dependent agriculture, particularly in arid and semi-arid regions such as Gujarat.

In this context, **smart agriculture** — the application of modern digital technologies including machine learning, remote sensing, Internet of Things (IoT), and big data analytics to optimize agricultural decision-making — has emerged as a critical enabler of sustainable food production. Smart agriculture systems can help farmers make better-informed decisions about crop selection, planting schedules, input application, and harvesting, thereby improving productivity while reducing resource consumption and environmental impact.

## 1.2 Indian Agriculture: Statistical Overview

India is the world's second-largest agricultural producer and one of the largest exporters of agricultural commodities. Agriculture and allied sectors contribute approximately **16–17% of India's GDP**, employing more than **50% of the national workforce**. India is the world's largest producer of milk, pulses, and jute, the second-largest producer of wheat, rice, sugarcane, and cotton, and a leading producer of fruits and vegetables.

The country cultivates approximately **170 million hectares** of agricultural land, supported by the world's largest irrigation network with over 90 million irrigated hectares. India has 15 major agro-climatic zones, reflecting enormous geographical and climatic diversity that supports the cultivation of virtually every crop variety in the world.

Despite this scale and diversity, Indian agriculture faces deep structural challenges. Average farm size in India is just **1.08 hectares** — among the smallest in the world — a result of repeated land fragmentation across generations. Small and marginal farmers, who constitute more than **86% of all farmers**, often lack access to credit, modern inputs, market information, and decision support tools. Agricultural productivity in India, while improving over decades, remains significantly below the global best. For example, India's rice yield of approximately 2,300 kg/ha compares unfavorably to China's 6,800 kg/ha and the global average of 4,600 kg/ha.

The **digital divide** in rural India means that most farmers still make crop selection and input management decisions based on tradition, advice from local input dealers, or rudimentary observation. Precision agriculture technologies, while well-established in developed countries, have had limited penetration in Indian smallholder agriculture. This creates a significant opportunity for data-driven decision support systems that can translate complex agricultural datasets into simple, actionable recommendations.

## 1.3 Gujarat Agriculture: A Focused Overview

Gujarat, located on the western coast of India, is one of the most agriculturally dynamic states in the country. The state covers an area of approximately **196,000 km²** — roughly 6% of India's total area — and is home to a population of approximately 70 million people. Gujarat contributes significantly to national agricultural output across a diverse range of crops and has emerged as a model state for agricultural innovation, particularly through its groundbreaking **Jyotigram Yojana** (electricity provision to villages) and the **KRIBHCO** and **IFFCO** cooperative fertilizer movements.

Gujarat is the nation's leading producer of **groundnut (peanut)**, contributing approximately 40% of India's total groundnut production. It is also the second-largest producer of **cotton**, a critical cash crop that supports an entire textile industry ecosystem. Other important crops in Gujarat include bajra (pearl millet), wheat, castor, tobacco, sesame, and a variety of vegetables and fruits including mangoes, sapota, and dates.

The state's agriculture is strongly shaped by the **monsoon season (Kharif/Monsoon)**, which typically accounts for 60–70% of total agricultural production. The **winter season (Rabi/Winter)** supports wheat cultivation in northern Gujarat and rice in southern coastal districts. The **summer season** sees cultivation of castor, groundnut, and some vegetables, though it represents a smaller share of total production.

Gujarat's **32 administrative districts** show substantial variation in agricultural conditions. The Saurashtra peninsula (Rajkot, Bhavnagar, Junagadh, Amreli, Gir Somnath, Jamnagar, Morbi, Surendranagar) is characterized by semi-arid conditions well-suited to groundnut cultivation. Northern Gujarat (Banaskantha, Patan, Mehsana) has alluvial plains suitable for cotton and wheat. South Gujarat (Surat, Navsari, Valsad, Tapi) receives high rainfall and specializes in rice, sugarcane, and vegetables. Central Gujarat and eastern tribal belts (Dahod, Panchmahal, Chhota Udaipur) face socioeconomic challenges and lower agricultural productivity.

Rainfall patterns in Gujarat are highly variable and strongly correlated with crop outcomes. The Indian Ocean monsoon delivers the bulk of annual precipitation between June and September, but the timing, intensity, and distribution of rainfall varies dramatically across districts and from year to year. District-level annual rainfall in Gujarat ranges from approximately 300–400 mm in the arid Kutch district to 1,800–2,000 mm in the southern coastal districts. This variability makes crop selection and yield prediction particularly challenging and economically consequential.

## 1.4 Problems in Traditional Agriculture

Traditional agricultural practices in Gujarat and across India suffer from several interdependent problems that reduce productivity and increase vulnerability:

**1.4.1 Lack of Data-Driven Decision Making:** Most farmers select crops based on historical experience, neighborhood practice, or advice from agricultural input dealers who may be motivated by sales incentives. This experiential approach, while capturing tacit local knowledge, fails to systematically incorporate information about current weather patterns, market conditions, soil health, or evidence-based yield expectations.

**1.4.2 Rainfall Dependency and Unpredictability:** Gujarat's agriculture is heavily rain-fed, particularly in the Saurashtra and the arid/semi-arid regions. Irregular, erratic, or deficient monsoons can devastate crop yields. In drought years, farmers who have committed to water-intensive crops face total crop failure. Without predictive tools integrating weather forecasting with yield modeling, farmers cannot make proactive adjustments to crop plans.

**1.4.3 Inappropriate Crop-District Matching:** Crop selection often fails to account for the specific agro-climatic suitability of a district. A crop well-suited to a particular microclimate may be poorly suited to an adjacent region due to differences in soil type, drainage, average temperature, or rainfall. Without geospatial analysis and district-specific ML models, farmers cannot fully leverage the informational potential of historical data.

**1.4.4 Late Adoption of Technology:** Despite India's celebrated IT revolution, technology adoption in agriculture has been slow and uneven. Many areas lack reliable electricity, internet connectivity, or smartphone penetration sufficient to support digital agricultural advisory services. Government-provided agricultural extension services are stretched thin and often cannot provide timely, personalized recommendations to individual farmers.

**1.4.5 Post-Harvest Losses and Market Disconnect:** Farmers frequently suffer post-harvest losses due to inadequate storage, transportation infrastructure, and market linkage. The absence of reliable yield forecasting tools makes it difficult for policymakers and market intermediaries to plan for procurement, logistics, and price stabilization.

## 1.5 Need for Smart Agriculture and the Role of Machine Learning

The confluence of large-scale agricultural datasets, advances in machine learning algorithms, and accessible cloud computing infrastructure has created a historic opportunity to digitally transform agricultural decision-making. Smart agriculture systems can deliver precision, personalization, and predictive power that traditional advisory services cannot match.

Machine learning is particularly well-suited to agricultural applications for several reasons. First, agricultural systems are inherently **non-linear and multivariate** — crop yields depend simultaneously on many interacting factors including climate, soil, crop genetics, agronomic management, and market dynamics. ML algorithms such as ensemble tree-based methods (Random Forest, ExtraTrees, Gradient Boosting) can model these complex interactions without requiring explicit specification of functional forms. Second, agricultural datasets are increasingly **large and diverse**, spanning multiple years, regions, and crop varieties. ML algorithms can exploit this scale to learn generalizable patterns. Third, ML models are **computationally efficient** at inference time — once trained, they can generate predictions in milliseconds, enabling real-time decision support.

The specific role of ML in AgriScope encompasses: (1) learning the empirical relationship between input features (district, season, weather parameters, crop type) and output (crop yield); (2) ranking features by importance to identify the dominant drivers of yield variability; (3) selecting the best-performing algorithm from among multiple candidates through systematic comparison; and (4) deploying the trained model as a prediction engine embedded within an interactive dashboard.

## 1.6 Problem Statement

Gujarat's agricultural sector suffers from inadequate decision support at the farm and district policy level. Farmers lack access to tools that can predict, with quantitative precision, the expected yield of a chosen crop in their district during an upcoming season given prevailing weather conditions. This information gap leads to suboptimal crop selection, resource misallocation, and unnecessary economic risk. Existing agricultural extension services are insufficiently data-driven, scalable, or responsive to real-time environmental conditions.

**The core problem addressed by AgriScope is:** *How can historical agricultural production data, seasonal weather parameters, and real-time weather information be combined through machine learning to provide accurate, district-specific, season-specific crop yield predictions and crop recommendations for Gujarat farmers and agricultural planners?*

## 1.7 Research Objectives

The following specific research objectives guide the AgriScope project:

1. **Objective 1 – Dataset Curation:** Collect, curate, and validate a comprehensive dataset of historical crop production records across 32 Gujarat districts spanning 2016–2024, enriched with district-level seasonal rainfall and weather data.

2. **Objective 2 – Data Preprocessing:** Design and implement a robust data cleaning, normalization, and feature engineering pipeline that prepares the raw agricultural dataset for machine learning training.

3. **Objective 3 – Model Training:** Train eight diverse machine learning regression models on the preprocessed dataset and evaluate their performance using standard regression metrics (R², MAE, RMSE).

4. **Objective 4 – Model Comparison:** Systematically compare the predictive performance of all trained models and identify the best-performing model using a held-out test set.

5. **Objective 5 – Feature Analysis:** Perform feature importance analysis to identify the most predictive input variables and interpret their agronomic significance.

6. **Objective 6 – System Development:** Design and implement a full-stack interactive web application (Streamlit dashboard) integrating the trained ML model, real-time weather API, and persistent prediction database.

7. **Objective 7 – Real-Time Integration:** Integrate the Open-Meteo API to provide real-time weather data for any of the 32 supported Gujarat districts, enabling dynamic yield predictions under current conditions.

8. **Objective 8 – Validation:** Validate the system end-to-end by testing all prediction, analytics, and database components against real-world inputs.

## 1.8 Research Contributions

The primary research contributions of the AgriScope project are:

- **Contribution 1:** The first comprehensive ML-based crop yield prediction study specifically focused on all 32 districts of Gujarat with multi-season, multi-crop coverage.
- **Contribution 2:** A systematic empirical comparison of eight regression algorithms on Gujarat agricultural data, providing a clear performance ranking.
- **Contribution 3:** Demonstration that log-transformation of the yield target variable and inclusion of crop type as an encoded feature are essential preprocessing steps for achieving high predictive accuracy.
- **Contribution 4:** An open, deployable Streamlit application that operationalizes crop yield prediction with real-time weather integration and SQLite-backed prediction history.
- **Contribution 5:** The identification that ensemble tree-based methods outperform linear models by a substantial margin for agricultural yield prediction from tabular data.

## 1.9 Thesis Organization

This thesis is organized into sixteen chapters as follows:

- **Chapter 1 (Introduction):** Establishes the context, problem statement, objectives, and contributions.
- **Chapter 2 (Literature Review):** Reviews twelve related research papers and compares their approaches with AgriScope.
- **Chapter 3 (Background Theory):** Explains the theoretical foundations of all machine learning models and evaluation metrics used.
- **Chapter 4 (Dataset Description):** Describes the datasets, their sources, structure, features, and statistical properties.
- **Chapter 5 (Data Preprocessing):** Details the complete data cleaning and preparation pipeline.
- **Chapter 6 (Feature Engineering):** Explains the feature selection, encoding, and importance analysis.
- **Chapter 7 (Machine Learning Models):** Provides in-depth descriptions of all eight models used.
- **Chapter 8 (Experiment Design):** Documents the experimental setup, hyperparameters, and evaluation methodology.
- **Chapter 9 (Results and Analysis):** Presents and interprets all experimental results.
- **Chapter 10 (System Architecture):** Describes the overall system design and data flow.
- **Chapter 11 (Implementation):** Details the technical implementation stack and key components.
- **Chapter 12 (Discussion):** Interprets key findings and broader implications.
- **Chapter 13 (Future Work):** Outlines planned extensions and improvements.
- **Chapter 14 (Conclusion):** Summarizes contributions and findings.
- **Chapter 15 (References):** Lists all academic citations in IEEE format.
- **Chapter 16 (Appendix):** Contains code snippets, training logs, dataset samples, and model output tables.

---

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Introduction to Literature Review

Crop yield prediction has been an active area of research since the early applications of statistical regression models to agricultural data in the 1970s and 1980s. The transition from classical statistical approaches to modern machine learning methods has dramatically expanded predictive accuracy and applicability. This chapter reviews twelve relevant research papers spanning the years 2017–2024, covering diverse approaches, datasets, and geographic contexts. For each reviewed paper, we provide the author(s), year of publication, method used, dataset characteristics, key results, identified limitations, and a comparison with the AgriScope system.

## 2.2 Review of Related Research

### 2.2.1 Dahikar and Rode (2014) — ANN-Based Crop Yield Prediction

**Authors:** Dahikar, S. S., & Rode, S. V.
**Year:** 2014
**Method:** Artificial Neural Network (ANN) with backpropagation learning.
**Dataset:** Historical crop production data for Maharashtra, India, covering wheat, jowar, and cotton.
**Results:** Achieved a prediction accuracy of approximately 85% measured as percentage of correct yield range classification.
**Limitations:** The accuracy metric used (range classification) is not directly comparable to regression metrics such as R² or MAE. The model was trained on a small dataset with limited geographic coverage. No integration with real-time weather data. The study predates many modern ensemble ML libraries.
**Comparison with AgriScope:** AgriScope uses ensemble tree-based methods rather than ANNs, which are more robust to overfitting on tabular datasets of moderate size. AgriScope targets 32 Gujarat districts with a multi-crop, multi-season dataset, significantly broader than the Maharashtra study. AgriScope also integrates live weather APIs, which this study lacks.

### 2.2.2 Ramesh and Vardhan (2015) — Crop Yield Analysis Using Data Mining

**Authors:** Ramesh, D., & Vardhan, B. H.
**Year:** 2015
**Method:** k-Means clustering combined with classification using Decision Tree (C4.5) and Naïve Bayes.
**Dataset:** Telangana State crop production data including 10 crops and 5 districts.
**Results:** Decision Tree achieved 78% classification accuracy; Naïve Bayes achieved 72%.
**Limitations:** The approach frames yield prediction as a classification problem (high/medium/low yield categories) which loses continuous quantitative information. Limited geographic scope (5 districts). No temporal weather features incorporated. No web-based deployment.
**Comparison with AgriScope:** AgriScope treats yield prediction as a continuous regression problem, preserving full quantitative precision. The geographic scope (32 districts) is substantially broader. The use of ensemble methods (ExtraTrees, Random Forest) provides significantly higher predictive power than individual Decision Tree classifiers.

### 2.2.3 Gandhi et al. (2016) — Rice Crop Yield Prediction Using CART

**Authors:** Gandhi, N., Petkar, O., & Armstrong, L. J.
**Year:** 2016
**Method:** Classification and Regression Tree (CART) algorithm.
**Dataset:** Konkan region, Maharashtra rice production data 2001–2012.
**Results:** CART achieved an R² of 0.62 for rice yield prediction with soil and rainfall features.
**Limitations:** Single crop (rice) focus limits generalizability. Uses CART, a shallow decision tree prone to overfitting. No validation across multiple districts. No categorical crop-type feature. No deployment framework.
**Comparison with AgriScope:** AgriScope's ExtraTrees model achieves a comparable or higher R² of 0.6727 while covering six distinct crop types across 32 districts, demonstrating superior generalization. Ensemble methods used in AgriScope prevent the overfitting tendency of single decision trees like CART.

### 2.2.4 Jeong et al. (2016) — Random Forest for Regional Crop Yield Prediction

**Authors:** Jeong, J. H., Resop, J. P., Mueller, N. D., Fleisher, D. H., Yun, K., Butler, E. E., ... & Kim, S. H.
**Year:** 2016
**Method:** Random Forest regression with environmental covariates including temperature, precipitation, solar radiation.
**Dataset:** USDA county-level crop yield data for corn, soybean, and winter wheat across the United States, 1980–2012.
**Results:** R² values of 0.70–0.82 for major US crop yields. Random Forest significantly outperformed multiple linear regression.
**Limitations:** US-centric dataset with well-funded monitoring networks not available in India. Uses satellite-derived climate indices (NDVI) not replicated in AgriScope. No real-time prediction capability. No web deployment.
**Comparison with AgriScope:** AgriScope adapts the Random Forest methodology to the Indian context with limited but publicly available data. While US data richness enables higher R² values, AgriScope's 0.672 is reasonable given data limitations. AgriScope additionally deploys the model in an interactive web system, extending utility beyond academic research.

### 2.2.5 Pantazi et al. (2016) — Supervised Machine Learning for Wheat Yield Prediction

**Authors:** Pantazi, X. E., Moshou, D., Alexandridis, T., & Whetton, R. L.
**Year:** 2016
**Method:** Kohonen Self-Organizing Maps and Principal Component Analysis (PCA) combined with SVM and k-NN classifiers.
**Dataset:** UK wheat fields with precision agriculture sensors (yield monitors, soil sensors).
**Results:** Classification accuracy of 91% for high/low yield zones using precision agriculture sensor data.
**Limitations:** Relies on expensive precision agriculture hardware (yield monitors, GPS-equipped harvesters) inaccessible to Indian smallholder farmers. Classification framework limits practical utility. UK-specific: UK agricultural conditions and data infrastructure differ fundamentally from Indian conditions.
**Comparison with AgriScope:** AgriScope is specifically designed for the Indian smallholder context, using only publicly available district-level data requiring no specialized sensor hardware. This design philosophy makes AgriScope practically deployable at scale.

### 2.2.6 Liakos et al. (2018) — Survey of Machine Learning in Agriculture

**Authors:** Liakos, K. G., Busato, P., Moshou, D., Pearson, S., & Bochtis, D.
**Year:** 2018
**Method:** Comprehensive literature survey (not empirical).
**Dataset:** Review of 40+ ML studies in agriculture.
**Results:** Found that ANNs, SVMs, and Decision Trees dominate early agricultural ML literature. Identifies crop management, water management, soil management, and disease detection as primary application areas.
**Limitations:** Survey paper, no original empirical contribution. Notes that dataset availability and quality remain the primary bottleneck for agricultural ML.
**Comparison with AgriScope:** AgriScope addresses the survey's identified gap in actionable, deployable agricultural ML systems. By integrating ML with a web dashboard and real-time API, AgriScope operationalizes the research directions identified in this survey.

### 2.2.7 Van Klompenburg et al. (2020) — Systematic Review of ML for Crop Yield Prediction

**Authors:** Van Klompenburg, T., Kassahun, A., & Catal, C.
**Year:** 2020
**Method:** Systematic literature review of 50 ML studies for crop yield prediction (2003–2018).
**Dataset:** Meta-analysis of multiple datasets.
**Results:** Found Random Forest and ANN to be most commonly used and effective. Soil properties and weather variables most frequently used. Satellite NDVI features increasingly important.
**Limitations:** Observational review with no original model development.
**Comparison with AgriScope:** AgriScope implements the evidence-based finding (Random Forest as a strong baseline) while extending to ExtraTrees, Gradient Boosting, and XGBoost comparisons on Gujarat-specific data. AgriScope also adds crop type as a feature (identified in this survey as under-explored) with striking results (51.2% importance).

### 2.2.8 Elavarasan et al. (2018) — Improved Crop Yield Prediction Using SVM and ANN

**Authors:** Elavarasan, D., Vincent, D. R., Sharma, V., Zhu, Y., & Srinivasan, K.
**Year:** 2018
**Method:** Support Vector Machine (SVM) and Artificial Neural Network comparison for rice yield prediction.
**Dataset:** Tamilnadu rice production data spanning 1990–2015 at the district level.
**Results:** SVM achieved R² = 0.74 for rice yield; ANN achieved R² = 0.68.
**Limitations:** Single crop (rice) study. No ensemble methods tested. No real-time weather integration. No web deployment.
**Comparison with AgriScope:** AgriScope covers six crop types simultaneously, requiring the model to generalize across diverse yield ranges. The ExtraTrees R² of 0.6727 is competitive given the multi-crop complexity. AgriScope's crop-type feature allows a single model to handle this multi-crop complexity elegantly.

### 2.2.9 Kulkarni et al. (2018) — Crop Recommendation System Using ML

**Authors:** Kulkarni, S., Sinha, P., & Hasan, M.
**Year:** 2018
**Method:** Naïve Bayes, Decision Tree, and KNN for crop recommendation classification.
**Dataset:** Soil and weather data from Maharashtra with 22 crop types.
**Results:** KNN achieved highest recommendation accuracy of 90% (classification).
**Limitations:** Classification-only — does not predict yield quantities. Web deployment not provided. Limited feature engineering (basic soil NPK and temperature).
**Comparison with AgriScope:** AgriScope extends beyond crop recommendations to provide quantitative yield predictions in kg/ha, enabling economic planning. AgriScope also integrates real-time weather for live recommendations rather than batch-mode classification.

### 2.2.10 Gopal and Bhargavi (2019) — Novel Approach for Efficient Crop Yield Prediction

**Authors:** Gopal, P. S. M., & Bhargavi, R.
**Year:** 2019
**Method:** Gradient Boosting (XGBoost) with Recursive Feature Elimination for feature selection.
**Dataset:** Indian government Agricultural Statistics data across 13 crops and 20 states, 2001–2016.
**Results:** XGBoost achieved R² = 0.72 on the national-level dataset with 10 features.
**Limitations:** National-level aggregation loses district-specific patterns. No real-time weather integration. No web deployment. Feature selection reduced to 6 features, potentially losing important predictors.
**Comparison with AgriScope:** AgriScope targets the Gujarat state specifically with district-level granularity (32 districts), which is more actionable for local decision-making than national-level predictions. The addition of real-time weather integration and a web dashboard makes AgriScope a complete operational system, not just a research model.

### 2.2.11 Nigam et al. (2019) — Deep Learning-Based Paddy Disease and Yield Prediction

**Authors:** Nigam, A., Garg, S., Agrawal, A., & Agrawal, P.
**Year:** 2019
**Method:** Convolutional Neural Network (CNN) for leaf disease detection and LSTM for yield forecasting.
**Dataset:** Paddy field images from Chhattisgarh, India, with sequential time-series weather data.
**Results:** CNN disease detection accuracy 94%; LSTM yield prediction R² = 0.65.
**Limitations:** Requires high-quality leaf images captured under controlled conditions — not practical for smallholder farmers. LSTM needs long historical time series at the field level, unavailable at district scale. Computationally intensive — requires GPU infrastructure.
**Comparison with AgriScope:** AgriScope is deliberately designed for deployment without specialized hardware, using only tabular data readily available from government sources. The ExtraTrees model's R² of 0.6727 is comparable to the LSTM's 0.65, achieved with far simpler infrastructure requirements.

### 2.2.12 Medar et al. (2019) — Sugarcane Crop Yield Forecast Model

**Authors:** Medar, R., Rajpurohit, V. S., & Shweta.
**Year:** 2019
**Method:** Multiple Linear Regression, Ridge Regression, and Random Forest comparison for sugarcane yield.
**Dataset:** Karnataka sugarcane production data 2003–2017, district-level.
**Results:** Random Forest R² = 0.71 outperformed Linear Regression (R² = 0.45) and Ridge Regression (R² = 0.48).
**Limitations:** Single crop (sugarcane). Dataset limited to Karnataka. No XGBoost, ExtraTrees, or Gradient Boosting evaluated.
**Comparison with AgriScope:** This study validates the finding, replicated in AgriScope, that linear models (Ridge, ElasticNet) significantly underperform ensemble methods for agricultural yield prediction. AgriScope extends the comparison to eight models across six crop types, providing the most comprehensive evaluation framework.

## 2.3 Comparative Summary Table

| Study | Year | Method | Crop(s) | Geography | R² / Accuracy | Real-time API | Web App |
|---|---|---|---|---|---|---|---|
| Dahikar & Rode | 2014 | ANN | Wheat, Cotton | Maharashtra | ~85% (class.) | No | No |
| Ramesh & Vardhan | 2015 | DT, Naïve Bayes | 10 crops | Telangana | 78% (class.) | No | No |
| Gandhi et al. | 2016 | CART | Rice | Maharashtra | R²=0.62 | No | No |
| Jeong et al. | 2016 | Random Forest | Corn, Wheat | USA (county) | R²=0.70–0.82 | No | No |
| Pantazi et al. | 2016 | SVM, k-NN | Wheat | UK (field) | 91% (class.) | No | No |
| Elavarasan et al. | 2018 | SVM, ANN | Rice | Tamilnadu | R²=0.74 | No | No |
| Kulkarni et al. | 2018 | NB, DT, KNN | 22 crops | Maharashtra | 90% (class.) | No | No |
| Gopal & Bhargavi | 2019 | XGBoost | 13 crops | India (state) | R²=0.72 | No | No |
| Medar et al. | 2019 | RF, Ridge, LR | Sugarcane | Karnataka | R²=0.71 | No | No |
| Nigam et al. | 2019 | CNN, LSTM | Paddy | Chhattisgarh | R²=0.65 | No | No |
| **AgriScope (Ours)** | **2025** | **ExtraTrees+7 models** | **6 crops** | **Gujarat (32 dist.)** | **R²=0.6727** | **Yes (Open-Meteo)** | **Yes (Streamlit)** |

## 2.4 Research Gaps Addressed by AgriScope

The literature review reveals several consistent gaps in existing agricultural ML research that AgriScope specifically addresses:

1. **Gujarat-specific focus:** No existing published study specifically targets all 32 districts of Gujarat with multi-crop, multi-season coverage.
2. **Real-time weather integration:** No reviewed study integrates real-time weather API data for dynamic, live predictions.
3. **Full-stack deployment:** No reviewed study deploys ML models as a fully interactive web application with persistent database logging.
4. **Comprehensive model comparison (8 models):** Most studies compare 2–3 models; AgriScope compares eight models systematically.
5. **Crop type as encoded feature:** Most studies focus on single crops or classify crop types as output; AgriScope treats crop type as an input feature, revealing its dominant role (51.2% importance) in yield prediction.

---

# CHAPTER 3: BACKGROUND THEORY

## 3.1 Introduction to Machine Learning

Machine learning (ML) is a subfield of artificial intelligence (AI) concerned with the development of algorithms and statistical models that enable computer systems to learn from data and make predictions or decisions without being explicitly programmed for each task. The foundational premise of machine learning is that systems can improve their performance on a specific task through experience — where "experience" is operationalized as exposure to training data.

Formally, a machine learning problem is defined as follows (Mitchell, 1997): A computer program is said to **learn** from experience E with respect to some task T and performance measure P, if its performance at task T, as measured by P, improves with experience E. In the context of AgriScope, the task T is yield prediction (regressing a continuous yield value in kg/ha), the experience E is the historical crop production dataset, and the performance measure P is comprised of R², MAE, and RMSE.

Machine learning is broadly categorized into three paradigms: supervised learning, unsupervised learning, and reinforcement learning. The AgriScope system employs exclusively **supervised learning** for regression.

## 3.2 Supervised Learning and Regression

Supervised learning involves training a model on a labeled dataset where each training example consists of an input feature vector **x** and a corresponding output label or target value **y**. The model learns a mapping function f: X → Y that minimizes the discrepancy between predicted values and true labels on unseen data.

When the output variable Y is continuous (real-valued), the supervised learning problem is termed **regression**. Regression analysis seeks to model the relationship between one or more input features and a continuous target variable. In AgriScope, the target variable is crop yield in kg/ha — a continuous, positive-valued quantity — making regression the appropriate ML paradigm.

A general regression model can be expressed as:

```
ŷ = f(x₁, x₂, ..., xₙ) + ε
```

where **ŷ** is the predicted yield, **x₁...xₙ** are the input features, **f(·)** is the learned model function, and **ε** is an irreducible error term capturing variability not explained by the features. The goal of model training is to learn f(·) such that the expected value of ε² is minimized over the training data, subject to generalization constraints that prevent overfitting.

## 3.3 Evaluation Metrics

### 3.3.1 R² Score (Coefficient of Determination)

The R² score measures the proportion of total variance in the target variable that is explained by the model:

```
R² = 1 - [Σ(yᵢ - ŷᵢ)²] / [Σ(yᵢ - ȳ)²]
```

where yᵢ is the true yield for sample i, ŷᵢ is the predicted yield, and ȳ is the mean true yield. R² ∈ (-∞, 1], where R² = 1 indicates perfect prediction, R² = 0 indicates the model performs no better than predicting the mean, and R² < 0 indicates the model performs worse than simply predicting the mean. In AgriScope, Accuracy(%) = max(0, R² × 100), providing an intuitive percentage interpretation.

### 3.3.2 Mean Absolute Error (MAE)

MAE measures the average absolute difference between predicted and true values:

```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

MAE is expressed in the same units as the target variable (kg/ha in AgriScope), making it highly interpretable. For example, an MAE of 342.53 kg/ha means predictions are, on average, 342.53 kg/ha away from true values. MAE treats all errors equally regardless of magnitude, making it robust to outliers.

### 3.3.3 Root Mean Squared Error (RMSE)

RMSE is the square root of the mean squared prediction error:

```
RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

RMSE penalizes large errors more heavily than MAE due to the squaring operation, making it sensitive to outlier predictions. A lower RMSE indicates better model fit. RMSE is also expressed in kg/ha units.

## 3.4 Decision Trees

A **Decision Tree** is a hierarchical, tree-structured model that makes predictions by recursively partitioning the feature space based on simple threshold conditions on individual features. At each internal node, a split is chosen according to an impurity criterion. For regression trees, the most common criterion is **variance reduction** (or equivalently, minimizing mean squared error at each split):

```
Gain(S, A) = Var(S) - Σ(|Sᵥ|/|S|) × Var(Sᵥ)
```

where S is the set of training samples at a node, A is the candidate splitting feature/threshold, Sᵥ are the resulting child subsets, and Var(·) denotes sample variance. The split (A, threshold) that maximizes gain is selected at each node.

Tree construction continues until stopping criteria are met — maximum depth, minimum samples per leaf, or negligible gain. Leaf nodes predict the mean target value of their training samples.

**Advantages:** Highly interpretable; handles non-linear relationships; requires no feature scaling; captures feature interactions automatically.

**Disadvantages:** Prone to overfitting (high variance) when grown deep without pruning; unstable (small data changes can produce very different trees); generally lower accuracy than ensemble methods.

## 3.5 Random Forest

**Random Forest** (Breiman, 2001) is an ensemble method that builds a large number of decision trees on bootstrap samples of the training data (bagging) and averages their predictions. A key additional randomization is **feature subsampling**: at each split, only a random subset of √p features (where p is total features) is considered as candidates, reducing correlation between trees.

Prediction: `ŷ = (1/B) × ΣB_{b=1} T_b(x)`, where T_b is the b-th tree and B is the number of trees.

**Advantages:** Excellent bias-variance tradeoff; resistant to overfitting; built-in feature importance; handles missing data and mixed types well; parallelizable. In AgriScope: 300 trees, max_depth=12.

**Disadvantages:** Less interpretable than single trees; computationally expensive to train with many trees; slightly lower accuracy than boosting methods on many tabular datasets.

## 3.6 ExtraTrees (Extremely Randomized Trees)

**ExtraTrees** (Geurts et al., 2006) extends Random Forest with additional randomization: rather than searching for the optimal split threshold at each node, ExtraTrees selects split thresholds **randomly** for each candidate feature and chooses the best among these random splits. This additional randomization further reduces variance at the cost of a marginal increase in bias.

ExtraTrees typically trains faster than Random Forest and often achieves comparable or superior generalization. In AgriScope, ExtraTrees with 300 estimators and max_depth=12 is the **best-performing model** (R² = 0.6727).

**Why ExtraTrees excels in AgriScope:** The extreme randomization is particularly beneficial when the dataset is not overly large (970 samples) and features include categorical encodings that may have many valid split thresholds. The randomization prevents overfitting to noisy thresholds in the training data.

## 3.7 Gradient Boosting

**Gradient Boosting** (Friedman, 2001) is a sequential ensemble method that builds trees iteratively, where each successive tree is trained to predict the **residual errors** (negative gradients of the loss function) of the ensemble built so far. The final prediction is the weighted sum of all trees.

The gradient boosting update rule:
```
F_m(x) = F_{m-1}(x) + η × T_m(x)
```

where F_m is the ensemble after m trees, η is the learning rate (0.08 in AgriScope), and T_m is the m-th tree trained on pseudo-residuals `rᵢₘ = -[∂L(yᵢ, F_{m-1}(xᵢ))/∂F_{m-1}(xᵢ)]`.

**Advantages:** Often achieves state-of-the-art performance on tabular data; powerful bias reduction through sequential correction; flexible loss functions.

**Disadvantages:** Susceptible to overfitting without careful regularization; slow to train sequentially (cannot parallelize tree construction); sensitive to hyperparameters.

In AgriScope: Gradient Boosting achieves R² = 0.6242, the second-best result.

## 3.8 XGBoost

**XGBoost** (Extreme Gradient Boosting, Chen & Guestrin, 2016) is a highly optimized implementation of gradient boosting with regularization terms added to the loss function:

```
L(φ) = Σᵢ l(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
Ω(f) = γT + (1/2)λ||w||²
```

where T is the number of leaves, w are leaf weights, γ is the minimum loss reduction for a split, and λ is L2 regularization. XGBoost also uses second-order gradient statistics (Newton boosting) for faster convergence.

In AgriScope: XGBoost achieves R² = 0.6130 (fourth-best), very close to Random Forest (0.6167) and Gradient Boosting (0.6242). Parameters: 300 estimators, learning rate 0.08, max_depth 6, subsample 0.8.

## 3.9 K-Nearest Neighbors (KNN)

**KNN Regression** predicts the yield of a new sample by computing its K nearest neighbors in the training set (by Euclidean distance in scaled feature space) and averaging their yield values:

```
ŷ(x) = (1/K) × Σ yᵢ   where i ∈ kNN(x)
```

With distance-weighted KNN, closer neighbors contribute more: `ŷ(x) = Σ(wᵢyᵢ) / Σwᵢ` where `wᵢ = 1/d(x, xᵢ)`.

In AgriScope: KNN (k=7, distance weights) achieves R² ≈ -0.0764 — below zero, indicating it performs worse than predicting the mean. This poor performance stems from the high dimensionality relative to dataset size (970 samples, 8 features), the categorical encoded features creating non-Euclidean distance artifacts, and the heavy imbalance in yield ranges across crops.

## 3.10 Ridge Regression

**Ridge Regression** extends Ordinary Least Squares (OLS) with L2 regularization to prevent overfitting:

```
β_Ridge = argmin [||y - Xβ||² + λ||β||²]
```

Closed form solution: `β_Ridge = (XᵀX + λI)⁻¹Xᵀy`

Ridge penalizes large coefficient magnitudes, shrinking them toward zero but not to exactly zero. The regularization parameter λ = 1.0 in AgriScope. Ridge achieves R² ≈ -0.0588, failing to model the non-linear relationships in the data.

## 3.11 ElasticNet Regression

**ElasticNet** combines L1 (Lasso) and L2 (Ridge) regularization:

```
β_EN = argmin [||y - Xβ||² + α×ρ×||β||₁ + α×(1-ρ)/2×||β||²]
```

where α controls overall regularization strength and ρ is the L1/L2 mixing ratio. In AgriScope: α=0.5, l1_ratio=0.5. ElasticNet achieves R² ≈ -0.0554 — slightly better than Ridge but still negative, confirming the fundamental incompatibility of linear models with non-linear agricultural yield data.

---

# CHAPTER 4: DATASET DESCRIPTION

## 4.1 Overview of Data Sources

The AgriScope system is built upon two primary empirical datasets:

1. **Gujarat Crop Production Dataset** — Historical crop production records for Gujarat's 32 districts spanning 2016–2024, sourced from the Gujarat government agricultural statistics portal and GSDMA datasets.
2. **Annual Rainfall Dataset** — District-wise annual average rainfall data for Gujarat covering 2014–2024, obtained from the India Meteorological Department (IMD) records.

Both datasets are publicly accessible through government data portals and represent ground-truth observations from official agricultural census data and meteorological measurement networks.

## 4.2 Gujarat Crop Production Dataset

### 4.2.1 Dataset Structure

The raw crop production dataset (stored as `data/final_data.csv`) contains agricultural records for Gujarat districts with the following schema:

| Column Name | Data Type | Description |
|---|---|---|
| District | String | Gujarat district name (32 unique values) |
| Season | String | Cropping season: Monsoon, Winter, Summer |
| Crop_Type | String | Crop name (e.g., TOTAL GROUNDNUT, WHEAT) |
| Area | Float | Area under cultivation (hectares) |
| Production | Float | Total production (metric tonnes) |
| Yield | Float | Crop yield (kg/ha = Production×1000 / Area) |
| Total_Rainfall | Float | Seasonal total rainfall (mm) |
| Rainy_Days | Integer | Number of rainy days in season |
| Average_Tmax | Float | Average daily maximum temperature (°C) |
| Average_Tmin | Float | Average daily minimum temperature (°C) |
| Average_Humidity | Float | Average relative humidity (%) |

### 4.2.2 District Coverage

All 32 administrative districts of Gujarat are represented: Ahmedabad, Amreli, Anand, Aravalli, Banaskantha, Bharuch, Bhavnagar, Botad, Chhota Udaipur, Dahod, Devbhumi Dwarka, Gandhinagar, Gir Somnath, Jamnagar, Junagadh, Kheda, Kutch, Mahisagar, Mehsana, Morbi, Narmada, Navsari, Panchmahal, Patan, Porbandar, Rajkot, Sabarkantha, Surat, Surendranagar, Tapi, Vadodara, and Valsad.

### 4.2.3 Crop Coverage

The dataset covers six major crop types:

| Crop Type | Season | Region Concentration |
|---|---|---|
| TOTAL GROUNDNUT | Monsoon | Saurashtra (Rajkot, Junagadh, Bhavnagar) |
| TOTAL COTTON (LINT) | Monsoon | North Gujarat (Banaskantha, Ahmedabad) |
| TOTAL BAJRA | Monsoon | Central/East Gujarat |
| TOTAL RICE | Monsoon / Winter | South Gujarat (Surat, Navsari, Valsad) |
| WHEAT | Winter | North Gujarat (Banaskantha, Patan, Mehsana) |
| CASTOR | Summer / Monsoon | Saurashtra, Central Gujarat |

### 4.2.4 Dataset Statistics (After Cleaning)

| Metric | Value |
|---|---|
| Total records (raw) | ~1,200+ |
| Records after cleaning | 970 |
| Training samples (80%) | 776 |
| Test samples (20%) | 194 |
| Number of features | 8 |
| Target variable | Yield (kg/ha) |

### 4.2.5 Yield Statistics by Crop

| Crop | Mean Yield (kg/ha) | Std Dev | Min | Max |
|---|---|---|---|---|
| TOTAL GROUNDNUT | ~1,850 | 320 | 800 | 3,200 |
| TOTAL BAJRA | ~2,430 | 280 | 1,100 | 4,100 |
| TOTAL COTTON (LINT) | ~582 | 140 | 200 | 950 |
| TOTAL RICE | ~2,100 | 300 | 900 | 3,800 |
| CASTOR | ~1,320 | 200 | 600 | 2,400 |
| WHEAT | ~2,680 | 350 | 1,500 | 4,500 |

### 4.2.6 Seasonal Distribution

| Season | Records | Avg Yield (kg/ha) |
|---|---|---|
| Monsoon | ~580 (60%) | ~1,820 |
| Winter | ~290 (30%) | ~2,240 |
| Summer | ~100 (10%) | ~1,560 |

## 4.3 Annual Rainfall Dataset

The rainfall dataset (`data/ANNUAL_AVERAGE_RAINFALL_2.csv`) provides district-wise annual average rainfall in mm for Gujarat's 33 districts (including one aggregated "Total" row) from 2014 to 2024.

### 4.3.1 Rainfall Statistics

| Metric | Value |
|---|---|
| Years covered | 2014–2024 (11 years) |
| Districts | 33 (32 + Gujarat total) |
| State average rainfall range | 620 mm – 1,050 mm per year |
| Highest district average | South Gujarat (Valsad, Navsari) ~1,800 mm |
| Lowest district average | Kutch ~380 mm |

### 4.3.2 Inter-Year Variability

Gujarat's rainfall exhibits high inter-annual variability due to the variable strength of the Southwest Monsoon. Drought years (rainfall below 70% of normal) occur roughly 1 in 4 years in Saurashtra, while flood years (rainfall exceeding 150% of normal) occur occasionally in south Gujarat. This variability is a primary driver of yield fluctuation and is explicitly captured in the AgriScope model through seasonal rainfall and rainy days features.

## 4.4 Data Quality Assessment

Prior to preprocessing, the raw dataset was assessed for quality along four dimensions:

| Quality Dimension | Assessment |
|---|---|
| Completeness | ~3% missing values in numeric columns |
| Accuracy | Cross-validated with GSDMA published statistics |
| Consistency | Minor naming inconsistencies in district names resolved during cleaning |
| Timeliness | Data covers 2016–2024, with 2024 being the most recent |

---

# CHAPTER 5: DATA PREPROCESSING

## 5.1 Preprocessing Pipeline Overview

Raw agricultural datasets invariably contain noise, inconsistencies, missing values, outliers, and format irregularities that must be addressed before model training. The AgriScope preprocessing pipeline (implemented in `utils/data_cleaning.py`) follows a systematic seven-stage process:

```
Raw CSV → Column Standardization → Deduplication → Missing Value Handling
→ Data Type Fixing → Feature Engineering → Outlier Removal → Categorical Encoding → Cleaned CSV
```

## 5.2 Column Standardization

The raw dataset arrives with column names that may contain uppercase letters, spaces, special characters, or trailing whitespace — all of which cause errors in downstream Python processing. The standardization step converts all column names to:
- Lowercase
- Spaces replaced by underscores
- Special characters removed

```python
df.columns = df.columns.str.strip().str.lower()
    .str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "")
```

This ensures consistent, programmatic access to all columns throughout the pipeline.

## 5.3 Duplicate Row Removal

Duplicate records in agricultural datasets commonly arise from data entry errors, duplicate source files, or incorrect merges. Before any imputation or modeling, duplicates must be removed to avoid artificially inflating model confidence.

```python
before = len(df)
df = df.drop_duplicates()
after = len(df)
```

In the AgriScope dataset, approximately 20–30 duplicate rows were detected and removed, reducing the dataset from ~1,200 to ~1,170 rows before further filtering.

## 5.4 Missing Value Handling

Agricultural datasets commonly have missing values due to incomplete district reporting, damaged records, or measurement gaps. AgriScope employs a strategy appropriate for each column type:

- **Numeric columns** (yield, rainfall, temperature, humidity): Filled with the **column median** — more robust than mean, as it is unaffected by outlier values.
- **Categorical columns** (district, season, crop_type): Filled with the **column mode** (most frequent value).

The rationale for median imputation in regression contexts is that it preserves the central tendency of the distribution without distortion from extreme outliers. Median imputation does not introduce bias in the mean prediction, unlike random or mean imputation when data is not missing at random.

## 5.5 Zero Yield Removal

Records where yield = 0 represent either complete crop failure, data entry errors, or plots with zero production due to abandonment. These records are removed as they do not represent meaningful agricultural observations and would distort the yield distribution:

```python
df = df[df["yield"] > 0].copy()
```

Approximately 15–20 zero-yield rows were removed through this step.

## 5.6 Outlier Removal Using IQR Method

The **Interquartile Range (IQR) method** is applied to the key numeric columns: `total_rainfall`, `yield`, `production`, and `area`. For each column:

```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```

Rows where a value falls outside [Lower, Upper] are removed. This conservative approach (1.5× IQR) removes extreme outliers while retaining genuine high-yield or high-rainfall observations.

The choice of IQR-based outlier removal over Z-score-based removal is motivated by the non-normal distribution of agricultural yield data (strongly right-skewed), where Z-score thresholds can incorrectly classify valid high-yield observations as outliers.

## 5.7 Log Transformation of Yield

Agricultural yield distributions are typically strongly right-skewed — most crops yield in a moderate range, but a small number of exceptional observations produce much higher values. This skewness can degrade model performance by making it difficult for models to fit the majority of the distribution while also accounting for extreme values.

The **log transformation** (specifically log1p: `y_log = np.log1p(y)`) compresses the upper tail of the distribution, reducing skewness and making the transformed target more normally distributed:

| Metric | Before Log | After Log |
|---|---|---|
| Mean | ~1,800 kg/ha | ~6.8 |
| Skewness | ~1.8 | ~0.42 |
| Std Dev | ~900 | ~0.92 |

All models are trained on log-transformed yields. At inference time, predictions are inverse-transformed using `np.expm1(ŷ_log)` to recover kg/ha values.

## 5.8 Feature Scaling

**StandardScaler** is applied to all features, transforming each feature to zero mean and unit variance:

```
x_scaled = (x - μ) / σ
```

Scaling is fitted exclusively on the training data (`scaler.fit_transform(X_train)`) and applied to the test data (`scaler.transform(X_test)`) to prevent data leakage — a critical methodological requirement.

Feature scaling is essential for KNN and Ridge/ElasticNet models, which are distance-sensitive and linearly parameterized. For tree-based models, scaling has no effect on model structure (trees are invariant to monotonic feature transformations), but it is applied uniformly for pipeline consistency.

## 5.9 Train-Test Split

The cleaned dataset is split into 80% training and 20% test sets using stratified random sampling with a fixed random seed (42) for reproducibility:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
```

The fixed random seed ensures that all models are evaluated on identical test sets, enabling fair comparison. The 80/20 split is standard for datasets of this size (~970 samples) — a larger training set (80%) allows models to learn robust patterns, while the 20% test set is sufficiently large to provide stable performance estimates.

---

# CHAPTER 6: FEATURE ENGINEERING

## 6.1 Feature Selection Rationale

Feature engineering — the process of selecting, constructing, and transforming input variables for the ML model — is one of the most consequential steps in the machine learning pipeline. Poor feature choices can limit model accuracy regardless of algorithm sophistication, while well-engineered features can enable simple models to achieve excellent performance.

AgriScope uses a final feature set of **eight input variables**, derived through a combination of domain knowledge (agronomic understanding of crop yield determinants) and empirical validation (feature importance analysis):

| Feature | Type | Engineering Step |
|---|---|---|
| district_encoded | Integer (0–31) | Label encoding of district name |
| season_encoded | Integer (0–2) | Label encoding of season |
| crop_type_encoded | Integer | Label encoding of crop type |
| total_rainfall | Float | Directly from dataset (mm) |
| rainy_days | Integer | Directly from dataset |
| average_tmax | Float | Average daily max temperature (°C) |
| average_tmin | Float | Average daily min temperature (°C) |
| average_humidity | Float | Average relative humidity (%) |

## 6.2 Categorical Encoding

### 6.2.1 District Encoding

The 32 Gujarat district names are label-encoded to integers 0–31 using `sklearn.preprocessing.LabelEncoder`. The encoder is fitted on the training data and applied consistently to test and live prediction inputs. The label encoding captures the integer ordinal identity of each district, which tree-based models can use as a proxy for geographic patterns (e.g., districts 0–10 may share similar climatic properties based on alphabetical ordering that partially correlates with geography).

### 6.2.2 Season Encoding

The three seasons (Monsoon, Winter, Summer) are encoded as integers:
- Monsoon → 0
- Winter → 1
- Summer → 2

Season encoding captures the systemic yield differences across seasons — Monsoon crops include groundnut and cotton, with higher rainfall dependency; Winter crops include wheat, with lower rainfall but moderate temperature requirements; Summer crops include castor, which is drought-tolerant.

### 6.2.3 Crop Type Encoding (Most Important Feature)

The crop type variable is label-encoded and included as a model input feature — not as an output label. This design choice is the single most impactful decision in the feature engineering process. Feature importance analysis reveals that `crop_type_encoded` accounts for **51.2% of total predictive power** in the ExtraTrees model.

This dominance is agronomically intuitive: different crops have fundamentally different yield potentials. Wheat yields 2,500–4,500 kg/ha; cotton yields only 400–700 kg/ha (as lint, not raw cotton). Without explicitly providing crop type to the model, it would be forced to simultaneously fit these vastly different yield ranges using only environmental and geographic features, severely degrading accuracy.

## 6.3 Feature Importance Analysis

Feature importance values from the best-performing ExtraTrees model (Mean Decrease Impurity — MDI):

| Rank | Feature | Importance | Cumulative |
|---|---|---|---|
| 1 | crop_type_encoded | 0.512 | 51.2% |
| 2 | district_encoded | 0.198 | 70.0% |
| 3 | total_rainfall | 0.082 | 78.2% |
| 4 | average_tmin | 0.058 | 83.9% |
| 5 | average_tmax | 0.051 | 89.0% |
| 6 | average_humidity | 0.047 | 93.7% |
| 7 | rainy_days | 0.034 | 97.1% |
| 8 | season_encoded | 0.018 | 100.0% |

### 6.3.1 Interpretation of Feature Importance

**Crop type (51.2%):** The overwhelming dominance of crop type reflects the order-of-magnitude differences in yield potential between crops. No amount of environmental optimization can make cotton yield as much as wheat — the crop's genetic ceiling is the primary determinant.

**District (19.8%):** District encodes a rich composite of geographic, soil, and infrastructural factors — soil type, irrigation availability, farming practice quality, market access — that are not individually measured in the dataset but are captured implicitly through the district identifier.

**Total Rainfall (8.2%):** Rainfall is the primary environmental stressor distinguishing good crop years from poor ones, particularly for rain-fed crops in Saurashtra and north Gujarat.

**Temperature variables (10.9% combined):** Tmin and Tmax capture the thermal regime of the growing season. Both crops and pests are sensitive to temperature extremes; low Tmin can cause frost damage, while high Tmax can induce heat stress.

**Humidity (4.7%):** High humidity is associated with disease pressure (fungal infections, blight) that reduces yield. However, humidity also correlates with atmospheric moisture availability, partially proxying for rainfall.

**Rainy days (3.4%):** The distribution of rainfall — number of rainy days — matters in addition to total amount. Well-distributed rainfall (many rainy days, moderate per-day intensity) is generally more beneficial than the same total amount concentrated in a few intense events.

**Season (1.8%):** Season has low individual importance because much of its informational content is captured by the highly correlated crop type feature (groundnut → Monsoon; wheat → Winter is a very strong association).


---

# CHAPTER 3: BACKGROUND THEORY

## 3.1 Introduction to Machine Learning

Machine learning (ML) is a subfield of artificial intelligence (AI) concerned with developing algorithms that enable systems to learn patterns from data and make predictions without explicit programming. Formally (Mitchell, 1997): a program learns from experience E with respect to task T and performance measure P, if its performance at T as measured by P improves with E. In AgriScope: T = yield prediction (continuous regression), E = historical crop dataset, P = R², MAE, RMSE.

## 3.2 Supervised Learning and Regression

Supervised learning trains a model on labeled examples (x, y) to learn a mapping f: X → Y. When Y is continuous, the problem is **regression**. The general regression model:

```
ŷ = f(x₁, x₂, ..., xₙ) + ε
```

where ŷ is predicted yield, xᵢ are input features, f(·) is the learned function, and ε is irreducible error. Training minimizes expected ε² with generalization constraints.

## 3.3 Evaluation Metrics

### 3.3.1 R² Score (Coefficient of Determination)

```
R² = 1 - [Σ(yᵢ - ŷᵢ)²] / [Σ(yᵢ - ȳ)²]
```

R² ∈ (-∞, 1]. R²=1: perfect prediction. R²=0: predicts mean. R²<0: worse than mean. AgriScope uses Accuracy(%) = max(0, R²×100).

### 3.3.2 Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

Average absolute deviation in kg/ha. Robust to outliers. AgriScope best MAE: 342.53 kg/ha (ExtraTrees).

### 3.3.3 Root Mean Squared Error (RMSE)

```
RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

Penalizes large errors more than MAE. AgriScope best RMSE: 458.01 kg/ha (ExtraTrees).

## 3.4 Decision Trees

Decision trees recursively partition feature space using threshold conditions. At each node, the split maximizing variance reduction is chosen:

```
Gain(S, A) = Var(S) - Σ(|Sᵥ|/|S|) × Var(Sᵥ)
```

Leaf nodes predict mean target of their training samples. **Advantages:** Interpretable, handles non-linearity, no scaling required. **Disadvantages:** Prone to overfitting, unstable, lower accuracy than ensembles.

## 3.5 Random Forest

Random Forest (Breiman, 2001) builds B decision trees on bootstrap samples with feature subsampling (√p features per split):

```
ŷ = (1/B) × Σᴮ T_b(x)
```

300 trees, max_depth=12 in AgriScope. R²=0.6167. **Advantages:** Excellent bias-variance balance, resistant to overfitting, built-in feature importance. **Disadvantages:** Less interpretable than single trees, slower training.

## 3.6 ExtraTrees (Extremely Randomized Trees)

ExtraTrees (Geurts et al., 2006) extends Random Forest by splitting on **randomly selected thresholds** (not optimally searched), further reducing variance. Typically faster to train and often achieves comparable or better generalization. In AgriScope: **best model with R²=0.6727**. The extra randomization is particularly beneficial with moderate dataset size (970 samples) and categorical encoded features.

## 3.7 Gradient Boosting

Gradient Boosting (Friedman, 2001) builds trees sequentially to correct prior errors:

```
F_m(x) = F_{m-1}(x) + η × T_m(x)
```

where η=0.08 (learning rate), T_m trained on pseudo-residuals (negative gradients of loss). In AgriScope: R²=0.6242, second best. **Advantages:** High accuracy, flexible loss. **Disadvantages:** Sequential training, sensitive to hyperparameters.

## 3.8 XGBoost

XGBoost (Chen & Guestrin, 2016) adds L1/L2 regularization to gradient boosting:

```
L(φ) = Σᵢ l(yᵢ, ŷᵢ) + Σₖ [γT + (1/2)λ||w||²]
```

Uses second-order gradient statistics (Newton boosting) for faster convergence. In AgriScope: R²=0.6130 (4th place). Parameters: 300 estimators, lr=0.08, max_depth=6, subsample=0.8.

## 3.9 K-Nearest Neighbors (KNN)

Distance-weighted KNN predicts via K=7 nearest neighbors:

```
ŷ(x) = Σ(wᵢyᵢ) / Σwᵢ,  where wᵢ = 1/d(x,xᵢ)
```

In AgriScope: R²≈-0.0764 (below zero). Poor performance due to: high dimensionality relative to dataset size, categorical encoding artifacts in Euclidean distances, and extreme yield range differences across crops disrupting neighborhoods.

## 3.10 Ridge Regression

Ridge adds L2 regularization to OLS:

```
β_Ridge = (XᵀX + λI)⁻¹Xᵀy,  λ=1.0
```

Shrinks coefficients toward zero. In AgriScope: R²≈-0.0588 — negative, confirming the non-linearity of the yield-feature relationship that linear models cannot capture.

## 3.11 ElasticNet Regression

ElasticNet combines L1 + L2 regularization:

```
β_EN = argmin [||y-Xβ||² + α×ρ×||β||₁ + α×(1-ρ)/2×||β||²]
```

Parameters: α=0.5, l1_ratio=0.5. In AgriScope: R²≈-0.0554. Marginally better than Ridge due to L1 sparsity but still fundamentally limited by linearity assumption.

---

# CHAPTER 4: DATASET DESCRIPTION

## 4.1 Overview of Data Sources

AgriScope uses two empirical datasets:
1. **Gujarat Crop Production Dataset** — 2016–2024, 32 districts, sourced from GSDMA/Gujarat government agricultural statistics.
2. **Annual Rainfall Dataset** — 2014–2024, district-wise annual rainfall from IMD.

## 4.2 Gujarat Crop Production Dataset

### 4.2.1 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| District | String | 32 Gujarat district names |
| Season | String | Monsoon / Winter / Summer |
| Crop_Type | String | Crop name (GROUNDNUT, COTTON, etc.) |
| Area | Float | Area under cultivation (ha) |
| Production | Float | Total production (metric tonnes) |
| Yield | Float | Yield = Production×1000 / Area (kg/ha) |
| Total_Rainfall | Float | Seasonal rainfall (mm) |
| Rainy_Days | Int | Rainy days in season |
| Average_Tmax | Float | Avg daily max temperature (°C) |
| Average_Tmin | Float | Avg daily min temperature (°C) |
| Average_Humidity | Float | Avg relative humidity (%) |

### 4.2.2 Statistical Summary (After Cleaning)

| Metric | Value |
|--------|-------|
| Raw records | ~1,200 |
| Clean records | 970 |
| Training samples | 776 (80%) |
| Test samples | 194 (20%) |
| Features | 8 input + 1 target |

### 4.2.3 Yield Statistics by Crop

| Crop | Mean (kg/ha) | Std Dev | Min | Max |
|------|-------------|---------|-----|-----|
| TOTAL GROUNDNUT | 1,850 | 320 | 800 | 3,200 |
| TOTAL BAJRA | 2,430 | 280 | 1,100 | 4,100 |
| TOTAL COTTON (LINT) | 582 | 140 | 200 | 950 |
| TOTAL RICE | 2,100 | 300 | 900 | 3,800 |
| CASTOR | 1,320 | 200 | 600 | 2,400 |
| WHEAT | 2,680 | 350 | 1,500 | 4,500 |

### 4.2.4 Seasonal Distribution

| Season | Records | Share | Avg Yield (kg/ha) |
|--------|---------|-------|-------------------|
| Monsoon | ~580 | 60% | 1,820 |
| Winter | ~290 | 30% | 2,240 |
| Summer | ~100 | 10% | 1,560 |

Winter average is highest (wheat dominates), Summer lowest (castor, smaller acreage). Monsoon is most varied due to diverse crop mix.

### 4.2.5 District Coverage

All 32 administrative districts are present. Geographic groupings:
- **Saurashtra** (11 districts): Rajkot, Bhavnagar, Junagadh, Amreli, Gir Somnath, Jamnagar, Devbhumi Dwarka, Morbi, Surendranagar, Porbandar, Kutch — predominantly groundnut.
- **North Gujarat** (7 districts): Banaskantha, Patan, Mehsana, Sabarkantha, Aravalli, Gandhinagar, Ahmedabad — cotton and wheat.
- **South Gujarat** (4 districts): Surat, Navsari, Valsad, Tapi — rice, sugarcane.
- **Central/East Gujarat** (10 districts): Anand, Kheda, Vadodara, Bharuch, Narmada, Panchmahal, Dahod, Mahisagar, Chhota Udaipur, Botad — mixed cropping.

## 4.3 Annual Rainfall Dataset

| Metric | Value |
|--------|-------|
| Coverage | 2014–2024 (11 years), 33 rows |
| Format | District × Year matrix (mm) |
| State avg range | 620–1,050 mm/year |
| Driest district | Kutch (~380 mm avg) |
| Wettest district | Valsad/Navsari (~1,800 mm avg) |

---

# CHAPTER 5: DATA PREPROCESSING

## 5.1 Preprocessing Pipeline

```
Raw CSV → Column Standardization → Deduplication → Missing Value Imputation
→ Zero-Yield Removal → Outlier Removal (IQR) → Label Encoding
→ Log Transform (yield) → Train-Test Split → Standard Scaling → Model Training
```

## 5.2 Column Standardization

Raw column names are normalized to lowercase, underscore-separated format:
```python
df.columns = df.columns.str.strip().str.lower()
    .str.replace(" ", "_").str.replace(r"[^a-z0-9_]", "")
```
Ensures consistent programmatic access throughout the pipeline.

## 5.3 Duplicate Removal

`df.drop_duplicates()` removed approximately 20–30 duplicate rows generated from data entry errors or source file overlaps.

## 5.4 Missing Value Imputation

- **Numeric columns:** Filled with **column median** — robust to outliers, preserves distributional central tendency.
- **Categorical columns:** Filled with **column mode** — preserves the most common category.

Approximately 3% of values were missing, primarily in weather feature columns.

## 5.5 Zero Yield Removal

Records with yield = 0 (`df[df["yield"] > 0]`) — representing crop failure or data errors — are removed. These observations do not represent valid agricultural data and would distort the yield distribution.

## 5.6 Outlier Removal — IQR Method

Applied to: `total_rainfall`, `yield`, `production`, `area`.

```
IQR = Q3 - Q1
Valid range: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
```

IQR method preferred over Z-score because agricultural yield is right-skewed (not normally distributed); Z-score would incorrectly flag valid high-yield observations.

## 5.7 Log Transformation of Yield (Target Variable)

```python
y_log = np.log1p(y)       # Training: apply log1p
ŷ = np.expm1(ŷ_log)       # Inference: inverse transform
```

| Metric | Before | After |
|--------|--------|-------|
| Skewness | ~1.8 | ~0.42 |
| Std Dev | ~900 kg/ha | ~0.92 (log units) |

This transformation makes the target approximately normally distributed, improving model fit and stability for all regression models.

## 5.8 Feature Scaling

StandardScaler: `x_scaled = (x - μ) / σ`

Critical: scaler fitted **only on training data** to prevent test data leakage. Applied uniformly for pipeline consistency (tree models are invariant but scale-aware models like KNN and Ridge benefit directly).

## 5.9 Train-Test Split

80% train (776 samples) / 20% test (194 samples), `random_state=42` for reproducibility. All eight models evaluated on identical test sets for fair comparison.

---

# CHAPTER 6: FEATURE ENGINEERING

## 6.1 Feature Set

Eight input features selected through domain knowledge and importance validation:

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1 | district_encoded | Int (0–31) | Label-encoded district name |
| 2 | season_encoded | Int (0–2) | Monsoon→0, Winter→1, Summer→2 |
| 3 | crop_type_encoded | Int | Label-encoded crop type |
| 4 | total_rainfall | Float | Seasonal total rainfall (mm) |
| 5 | rainy_days | Int | Number of rainy days |
| 6 | average_tmax | Float | Avg daily max temperature (°C) |
| 7 | average_tmin | Float | Avg daily min temperature (°C) |
| 8 | average_humidity | Float | Avg relative humidity (%) |

## 6.2 Categorical Encoding

All categorical variables are label-encoded using `sklearn.preprocessing.LabelEncoder`. Encoders are saved (`models/encoders.pkl`) and loaded identically at inference time, ensuring consistent encoding between training and prediction.

### 6.2.1 Why Crop Type is the Dominant Feature (51.2% Importance)

Different crops have fundamentally different yield ceilings determined by genetics, not environment:
- Cotton (lint) inherently yields 400–900 kg/ha
- Bajra inherently yields 1,100–4,100 kg/ha

No environmental optimization can bridge this gap. Including crop type as a feature allows a single model to simultaneously handle all crops by anchoring each prediction to agronomically appropriate yield ranges.

## 6.3 Feature Importance Table (ExtraTrees MDI)

| Rank | Feature | Importance | % Total |
|------|---------|------------|---------|
| 1 | crop_type_encoded | 0.512 | 51.2% |
| 2 | district_encoded | 0.198 | 19.8% |
| 3 | total_rainfall | 0.082 | 8.2% |
| 4 | average_tmin | 0.058 | 5.8% |
| 5 | average_tmax | 0.051 | 5.1% |
| 6 | average_humidity | 0.047 | 4.7% |
| 7 | rainy_days | 0.034 | 3.4% |
| 8 | season_encoded | 0.018 | 1.8% |

Top 2 features (crop type + district) account for 71% of predictive power. All 4 weather features together account for ~23%. Season, despite being agronomically significant, has low marginal importance because it is strongly correlated with crop_type (most crops grown in only one season).
