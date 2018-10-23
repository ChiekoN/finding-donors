
# Project: Finding Donors for CharityML
## Category: Supervised Learning


### Project Overview

In this project, I applied supervised learning techniques on data collected for the U.S. census to help CharityML (a fictitious charity organization) identify people most likely to donate to their cause.

I first explored the data to learn how the census data is recorded. Next, I applied a series of transformations and preprocessing techniques to manipulate the data into a workable format. Then I evaluated three types of supervised learners (Random Forest, SVMs, and Logistic Regression) on the data, compared performances, and chose one of those models with a reasonable explanation. Afterwards, I optimized the model and present it as my solution to CharityML. Finally, I explored the chosen model and its predictions under the hood, to see just how well it's performing when considering the data it's given.

### Setting

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. My goal is to build an algorithm to best identify potential donors for the charity, and help them raise funds effectively with reducing overhead cost of sending mail.

### Techniques

- Data preprossessing
  - Transforming skewed continuous features using logarithm
  - Normalization (scaling)
  - One-hot encoding
- Model evaluation
  - Accuracy
  - Recall and precision
  - F-beta score
- Supervised learning models
  - Random Forest
  - Support Vector Machines
  - Logistic Regression
- Model optimization by tuning hyperparameters
- How to train and test a supervised learning model
- Feature selection

### Libraries

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

This project runs on the [Jupyter Notebook](http://ipython.org/notebook.html)

### Files

- `finding_donors.ipynb`: Jupyter notebook, This project's main file
- `census.csv`: CSV file, The project data file
- `visuals.py`: Python, visualization code
- `finding_donors.html`: HTML, Output of the project (for submission)

### Data

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)


### Note

This project has been done as a part of Machine Learning Engineer Nanodegree program, at [Udacity](https://www.udacity.com/).
