# Linear Regression: Predicting House Prices from First Principles

This repository builds linear regression from the ground up, without relying on a fitting library. Starting from a nine-point toy dataset, you compute expected value, variance, and covariance by hand using NumPy, then derive the slope and intercept of a single-variable predictor analytically. You then extend that idea to multiple input features by constructing a design matrix and solving the normal equation with matrix algebra. Every step uses a concrete real estate dataset - predicting house prices per unit area - so each formula has an immediately interpretable economic meaning. A companion homework notebook applies the identical pipeline to a second dataset so you can confirm understanding without a worked key in front of you.

---

## Learning Objectives

- Compute expected value, variance, and covariance from raw data using NumPy
- Explain why dividing by n-1 (not n) gives an unbiased sample estimate of covariance and variance
- Derive the slope and intercept of a simple linear predictor analytically from covariance and variance
- Interpret a negative slope economically (e.g., older houses tend to be cheaper, longer MRT distance lowers price)
- Construct a multi-variable predictor using the normal equation: W = (X^T X)^{-1} X^T Y
- Build a design matrix by stacking feature columns and understand why the column ordering must match the weight ordering
- Recognize when a single-variable predictor is insufficient and explain why adding correlated features changes existing weight estimates

---

## Data / File Dictionary

| File | Description |
|------|-------------|
| `linear_regression_lesson.ipynb` | Fully worked lesson notebook covering Parts 0-2: summary statistics on a toy dataset, single-variable predictor on house age, and multi-variable predictor via the normal equation using the Taiwanese real estate dataset |
| `linear_regression_homework.ipynb` | Homework notebook with `...` placeholders for student completion; applies the same three-part pipeline to the California housing dataset from scikit-learn |
| `real_estate_data.csv` | 414 real estate transactions from New Taipei City, Taiwan - features include house age, MRT distance, convenience store count, latitude, and longitude, with price per unit area as the target |
| `requirements.txt` | Python package dependencies (numpy, pandas, scikit-learn) needed to run both notebooks |

### Dataset columns - lesson notebook (real_estate_data.csv)

| Column | Meaning |
|--------|---------|
| `X1 house age` | Age of the house in years |
| `X2 distance to the nearest MRT station` | Distance to the nearest transit station in meters |
| `X3 number of convenience stores` | Count of nearby convenience stores |
| `X4 latitude` | Latitude coordinate |
| `X5 longitude` | Longitude coordinate |
| `Y house price of unit area` | House price per unit area (target variable) |

### Dataset columns - homework notebook (California housing, sklearn built-in, first 414 rows)

| Column | Meaning |
|--------|---------|
| `MedInc` | Median income of the block group in tens of thousands of USD |
| `HouseAge` | Median house age of the block group in years |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `MedHouseVal` | Median house value in hundreds of thousands of USD (target variable) |

---

## Workflow Diagram

```
Lesson: real_estate_data.csv              Homework: sklearn fetch_california_housing()
         |                                           |  (first 414 rows)
         v                                           v
 [ Load with pandas ]                     [ Load with sklearn ]
         |                                           |
   ------+----------------------------------   ------+-----------------------------------
   |                                      |   |                                        |
   v                                      v   v                                        v
Part 0: Toy dataset (9 points)     Part 1: Single-variable        Sec 1: Summary statistics
  E[X], E[Y], Cov(X,Y), E[X^2]    predictor on house age          on median income
  Verify: Var(X) = E[X^2]-(E[X])^  slope = Cov(X1,Y) / Var(X1)
                                   predictor(age) -> price        Sec 2: Single-variable
                                           |                       predictor on income
                                           v
                                   Part 2: Multi-variable         Sec 3: Multi-variable
                                   predictor via normal eq         predictor via normal eq
                                   X = [age | MRT dist | stores]  X = [income | rooms | age]
                                   W = (X^T X)^{-1} X^T Y
```

---

## Step-by-Step Walkthrough

### Part 0 - Summary statistics from scratch

Before reaching for a library, the lesson computes E[X], E[Y], Cov(X, Y), and E[X^2] on a nine-point toy dataset with intentionally high noise. The noise is deliberate: this part is about practicing the formulas, not building an accurate model.

The payoff of computing by hand is understanding what `np.cov` actually returns. It gives a 2x2 matrix; the off-diagonal entry [0][1] is Cov(X, Y). Knowing this prevents a common mistake of using the whole matrix as a scalar.

The part ends with a concept check: verify the algebraic identity Var(X) = E[X^2] - (E[X])^2 numerically, using both the formula and `np.var(x, ddof=0)`. This identity links the two formulas for variance and is worth internalizing because it appears in derivations of the normal equation.

### Part 1 - Single-variable predictor

With the real estate dataset loaded (414 Taiwanese transactions), the lesson predicts house price per unit area from house age alone. The closed-form solution is:

```
slope     = Cov(X1, Y) / Var(X1)
intercept = E[Y] - slope * E[X1]
predictor(x) = intercept + slope * x
```

The slope is computed using sample statistics with ddof=1 (n-1 denominator) to get an unbiased estimate from the 414-point sample rather than a biased population estimate. Using ddof=0 here would slightly underestimate true variance and slightly overstate the slope magnitude.

The slope turns out negative: older houses have lower predicted prices on average. This is economically intuitive - newer construction commands a premium in this market. The concept check asks you to compute the slope for MRT distance (x2) and interpret its sign, reinforcing that the formula gives you a direction, not just a magnitude.

Crucially, the predictor function closes over the slope and intercept computed from the summary statistics. This makes the chain of dependencies explicit: a wrong Cov or Var flows directly into every prediction. It is a design decision to show the full error-propagation path before using a library that hides it.

### Part 2 - Multi-variable predictor via the normal equation

One feature is rarely enough. Part 2 adds MRT distance (x2) and convenience store count (x3). With three inputs, the scalar formula no longer applies and you need the normal equation:

```
W = (X^T X)^{-1} X^T Y
```

The design matrix X is built with `np.column_stack`, producing a (414, 3) array where each row is one house and each column is one feature. Y is reshaped to a (414, 1) column vector so the matrix multiplication is well-defined. The weight vector W is then solved in four explicit steps - transpose, multiply, invert, multiply again - rather than in a single call, so each matrix operation maps to one term in the formula.

The weight signs are meaningful: w2 (MRT distance) should be negative (farther from transit means lower price) and w3 (convenience stores) should be positive. Reading weights with their sign and units is how you connect a matrix solution back to real-world intuition.

The concept check extends this to four features by adding latitude (x4), and asks whether the existing weights change. They will, because features are not perfectly independent. This is a preview of multicollinearity - a concept that becomes important in larger regression problems.

### Homework - applying the same pipeline to California housing

The homework mirrors Parts 0-2 exactly but uses the California housing dataset from scikit-learn, trimmed to 414 rows to keep the dataset size comparable. The features are median income, average rooms, and median house age; the target is median house value in hundreds of thousands of USD.

Working through the homework without the lesson solution visible confirms that the skill transfers to a new dataset with different units, different feature scales, and a different economic story (block-level census aggregates vs. individual transactions).

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/ehcastroh-teach/Linear_Regression.git
cd Linear_Regression

# Install dependencies
pip install -r requirements.txt

# Launch the lesson notebook (fully worked)
jupyter notebook linear_regression_lesson.ipynb

# Or open the homework notebook (fill in the ... placeholders)
jupyter notebook linear_regression_homework.ipynb
```

Work through each notebook from top to bottom. Each question builds on the variables computed in the previous one - for example, the slope in Part 1 uses the covariance and variance you computed in Part 0. Fill in `...` placeholders with your own code and run the cell to check the result. The concept checks at the end of each part are optional extensions that go slightly beyond the guided questions.

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| Expected value (E[X]) | The arithmetic mean of a variable - the average outcome if you sampled infinitely many times from the same distribution |
| Variance (Var(X)) | How spread out a variable is around its mean; computed as E[X^2] - (E[X])^2, or equivalently as the average squared deviation from the mean |
| Covariance (Cov(X, Y)) | How much two variables move together; positive when they tend to rise and fall together, negative when they move in opposite directions |
| Sample vs. population statistic | Sample statistics use n-1 in the denominator (ddof=1) to give an unbiased estimate of the true population parameter from a finite sample |
| Simple linear regression | Fitting a straight line Y = a + bX by choosing slope b and intercept a to minimize the sum of squared prediction errors |
| Slope | The change in predicted Y for a one-unit increase in X; equals Cov(X, Y) / Var(X) |
| Intercept | The predicted value of Y when X equals zero; a mathematical anchor, not always a realistic input value |
| Normal equation | The closed-form solution W = (X^T X)^{-1} X^T Y that gives the exact least-squares weights without iteration |
| Design matrix | The matrix X where each row is one observation and each column is one input feature; column ordering must match the weight ordering in W |
| Weight (coefficient) | A single entry in W; scales one input feature and has the same sign and direction of effect as the slope in simple regression |
| Residual | The difference between the actual target value and the predicted value for one observation |
| Least squares | The objective of minimizing the sum of squared residuals across all data points; what the normal equation solves exactly |
| Transpose (X^T) | Flipping a matrix so rows become columns; required to form X^T X and X^T Y in the normal equation |
| Matrix inverse | The matrix equivalent of division; (X^T X)^{-1} exists when the columns of X are linearly independent |
| Multicollinearity | When two or more input features are highly correlated with each other; causes weight estimates to shift when features are added or removed |

---

## Further Reading

- "An Introduction to Statistical Learning" - Chapters 2-3
- "The Elements of Statistical Learning" - Chapter 3
- "Python for Data Analysis" - NumPy chapter
- "Mathematics for Machine Learning" - Chapter 9 (linear regression)
- NumPy linear algebra reference: numpy.linalg module
- scikit-learn user guide: California housing dataset

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
