\# Electricity Demand Forecasting



A reproducible time‑series forecasting pipeline for modelling and forecasting German electricity demand.



\## Project Aim



The aim of this project is to forecast weekly German electricity demand and compare the performance, interpretability, and complexity of different forecasting approaches.



The main research questions are:



\- How well do simple benchmark methods forecast weekly electricity demand?

\- Does a SARIMAX model improve on seasonal benchmarks?

\- Do temperature and holiday covariates improve forecast accuracy?

\- Do feature‑based or neural models justify their additional complexity?

\- Which model would be most appropriate for an operational forecasting setting?



\## Data



The target series is German electricity load from \[Open Power System Data (OPSD)](https://data.open-power-system-data.org/time\_series/). The original data are hourly electricity load observations, cleaned, aggregated to weekly average load, and converted from MW to GW.



Optional covariates include:



\- `temp\_mean`, `temp\_min`, `temp\_max`

\- `heating\_degree\_days`, `cooling\_degree\_days`

\- `holiday\_days`, `has\_holiday`



Temperature features are external covariates and should be treated carefully. In a real operational setting, future temperature would not be known exactly and would need to come from a weather forecast. If realised future temperature is used in the test period, the resulting forecast is described as a \*conditional forecast\*.



Holiday features are generally known in advance and are valid future covariates.



\## Repository Structure

electricity-demand-forecasting/

│

├── README.md

├── requirements.txt

├── environment.yml

├── .gitignore

│

├── data/

│ ├── raw/

│ ├── interim/

│ └── processed/

│

├── src/

│ └── electricity\_demand/

│ ├── init.py

│ ├── config.py

│ ├── pipeline.py

│ ├── data.py

│ ├── features.py

│ ├── evaluation.py

│ ├── plotting.py

│ └── models/

│ ├── init.py

│ ├── benchmarks.py

│ ├── sarimax.py

│ ├── feature\_models.py

│ ├── bayesian.py

│ └── neural.py

│

├── scripts/

│ ├── download\_data.py

│ ├── make\_features.py

│ ├── run\_pipeline.py

│ └── evaluate\_models.py

│

├── outputs/

│ ├── figures/

│ ├── forecasts/

│ ├── metrics/

│ └── model\_objects/

│

├── reports/

│ ├── report.md

│ └── figures/

│

└── tests/

├── test\_benchmarks.py

├── test\_evaluation.py

└── test\_features.py

