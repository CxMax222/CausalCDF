# Modeling Question Difficulty for Unbiased Cognitive Diagnosis: A Causal Perspective

Source code for the paper *Modeling Question Difficulty for Unbiased Cognitive Diagnosis: A Causal Perspective*.

The code is the implementation of CausalNCDM model.


## Dependencies:

- python 3.6
- pytorch >= 1.0
- numpy
- csv
- pandas
- sklearn



## Usage

Run run.py train the model and test the trained the model on the test set:

`python run.py`




## Data Set

The data is extracted from public data set [ASSIST2009-2010](https://sites.google.com/site/assistmentsdata/home/assistment2009-2010-data/skill-builder-data-2009-2010) (skill-builder, corrected version) where `answer_type != 'open_response' and skill_id != ''`. When a user answers a problem for multiple times, only the first time is kept.
We split the data into training dataset, validation dataset and testing dataset.




## Details

- We filter students with less than 15 answering logs.
- The model parameters are initialized with Xavier initialization.
- The model uses Adam Optimizer, and the learning rate is set to 0.001.
- We set the default values at γ = 0.1.


