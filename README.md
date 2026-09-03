# Linear Regression

This repository walks through linear regression from first principles, building up from computing means and covariances by hand to constructing both single-variable and multi-variable predictors using matrix algebra. The dataset is a real estate pricing table, giving every calculation a concrete, interpretable meaning: you are predicting house prices, not working through abstract math.

---

## Learning Objectives

- Compute expected value, variance, and covariance from raw data using NumPy
- Derive the slope and intercept of a simple linear predictor from covariance and variance
- Construct a multi-variable predictor using the normal equation: W = (X^T X)^{-1} X^T Y
- Interpret regression weights in plain language against a real dataset
- Recognize when a single-variable predictor is insufficient and why adding features helps

---

## Data / File Dictionary

| File | Description |
|------|-------------|
| `linear_regression_lesson.ipynb` | Lesson notebook with full worked solutions covering Parts 0-2: summary statistics, single-variable regression, and multi-variable regression using a Taiwanese real estate dataset |
| `linear_regression_homework.ipynb` | Homework notebook with `...` placeholders for student completion, applied to the California housing dataset from scikit-learn |
| `real_estate_data.csv` | 414 real estate transactions with house age, MRT distance, convenience store count, latitude, longitude, and price per unit area (used in the lesson) |
| `requirements.txt` | Python package dependencies needed to run the notebooks |

### Dataset columns (lesson notebook - real_estate_data.csv)

| Column | Meaning |
|--------|---------|
| `X1 house age` | Age of the house in years |
| `X2 distance to the nearest MRT station` | Distance to nearest transit station in meters |
| `X3 number of convenience stores` | Count of nearby convenience stores |
| `X4 latitude` | Latitude coordinate |
| `X5 longitude` | Longitude coordinate |
| `Y house price of unit area` | House price per unit area (target variable) |

### Dataset columns (homework notebook - California housing, sklearn built-in)

| Column | Meaning |
|--------|---------|
| `MedInc` | Median income of block group (in tens of thousands of USD) |
| `HouseAge` | Median house age of block group in years |
| `AveRooms` | Average number of rooms per household |
| `MedHouseVal` | Median house value (target, in hundreds of thousands of USD) |

---

## Workflow Diagram

```
Lesson: real_estate_data.csv          Homework: sklearn fetch_california_housing()
        |                                         |
        v                                         v
[ Load with pandas ]                    [ Load with sklearn ]
        |                                         |
   -----+-------------------------------     -----+-------------------------------
   |                                   |    |                                    |
   v                                   v    v                                    v
Part 0: Toy dataset              Part 1: Single-variable      Sec 1: Summary stats
(9 points, inline)               predictor on house age        on median income
  E[X], E[Y], Cov, E[X^2]         slope = Cov / Var
                                     predictor(age) -> price   Sec 2: Single-variable
                                          |                     predictor on income
                                          v
                                  Part 2: Multi-variable       Sec 3: Multi-variable
                                  predictor via normal eq       predictor via normal eq
                                  W = (X^T X)^{-1} X^T Y       (income, rooms, age)
```

---

## Step-by-Step Walkthrough

### Part 0 - Summary statistics from scratch

Before reaching for a library, you compute the building blocks by hand on a 9-point toy dataset. This forces familiarity with what `np.mean`, `np.cov`, and `np.var` actually calculate, which matters when interpreting model output later.

The key quantity here is `Cov(X, Y)` - it captures how much X and Y move together. If it is positive, higher X tends to mean higher Y. If negative, they move in opposite directions. Linear regression is essentially a normalized version of this idea.

### Part 1 - Single-variable predictor

With the real estate dataset loaded, you predict house price from house age alone. The formulas are:

```
slope     = Cov(X, Y) / Var(X)
intercept = E[Y] - slope * E[X]
predictor(x) = intercept + slope * x
```

This is the closed-form solution for simple linear regression. No gradient descent, no optimization library - just two statistics from the data. You derive the slope from your covariance and variance answers, which means any error in those carries forward. This is intentional: it shows that a regression model is only as reliable as the summary statistics feeding it.

### Part 2 - Multi-variable predictor via the normal equation

One variable is rarely enough. Part 2 adds MRT distance and convenience store count. With multiple inputs, the scalar formula no longer applies - you need matrix algebra.

The normal equation solves for the weight vector W that minimizes squared prediction error:

```
W = (X^T X)^{-1} X^T Y
```

This is the exact, analytic solution. You build X as a matrix where each row is one house and each column is one feature, then apply the formula with `np.linalg.inv` and `.dot`. The result is a vector of three weights - one per feature - telling you how much each input contributes to the predicted price.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/ehcastroh-teach/Linear_Regression.git
cd Linear_Regression

# Install dependencies
pip install -r requirements.txt

# Launch the lesson notebook (worked solutions)
jupyter notebook linear_regression_lesson.ipynb

# Or open the homework notebook (fill in the blanks)
jupyter notebook linear_regression_homework.ipynb
```

Work through the notebook top to bottom. Each question builds on the previous one. Fill in the `...` placeholders with your own code and run each cell to check your result.

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| Expected value (E[X]) | The mean of a variable - the average outcome if you sampled infinitely many times |
| Variance (Var(X)) | A measure of how spread out a variable is around its mean |
| Covariance (Cov(X, Y)) | A measure of how much two variables change together; positive means they tend to rise and fall together |
| Simple linear regression | Fitting a straight line Y = a + bX to data by choosing slope b and intercept a to minimize error |
| Slope | The change in the predicted Y for a one-unit increase in X; equals Cov(X, Y) / Var(X) |
| Intercept | The predicted value of Y when X equals zero |
| Normal equation | The closed-form solution W = (X^T X)^{-1} X^T Y that gives the exact least-squares weights without iteration |
| Weight matrix (W) | The vector of coefficients in a multi-variable predictor; each weight scales one input feature |
| Feature | An input variable used to make a prediction (house age, MRT distance, etc.) |
| Residual | The difference between the actual value and the predicted value for a given data point |
| Least squares | The objective of minimizing the sum of squared residuals across all data points |
| Transpose (X^T) | Flipping a matrix so rows become columns; required to form X^T X in the normal equation |
| Matrix inverse | The matrix equivalent of division; (X^T X)^{-1} exists when the columns of X are linearly independent |
| Design matrix | The matrix X where each row is one observation and each column is one input feature |

---

## Further Reading

- "An Introduction to Statistical Learning" - Chapters 2-3 (linear regression)
- "The Elements of Statistical Learning" - Chapter 3 (linear methods for regression)
- "Python for Data Analysis" - Chapter on NumPy and array operations
- NumPy linear algebra documentation: `numpy.linalg` module reference
- scikit-learn California housing dataset documentation: `sklearn.datasets.fetch_california_housing`

---

## Credits and Acknowledgements

Lesson dataset derived from a publicly available real estate transaction dataset covering transactions in the New Taipei City area of Taiwan.

Homework dataset: California housing data from the StatLib repository, included with scikit-learn. Originally published in Pace, R.K. and Barry, R. (1997) "Sparse Spatial Autoregressions", Statistics and Probability Letters, 33(3), 291-297.

---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
