### Introduction

In this scenario, we're going to look at what predictive modeling is and how it uses statistics to predict outcomes from existing data. We'll cover real world examples to understand the concepts better. We'll see what regression analysis means and analyze some of its forms in detail. We'll also look at an example which predicts the price of a car for us.

These are the topics that we'll cover in this scenario:

- Linear regression and how to implement it in Python
- Polynomial regression, its application and examples
- Multivariate regression and how to implement it in Python
- An example we'll build that predicts the price of a car using Python
- The concept of multi-level models and some things to know about them

### Jupyter Notebooks

We will run Jupyter Notebook as a Docker container. This setup will take some time because of the size of the image.

## Login
When the container is running, execute this statement:
`docker logs jupyter 2>&1 | grep -v "HEAD" `


This will show something like:
```
copy/paste this URL into your browser when you connect for the first time, to login with a token:
    http://localhost:8888/?token=f89b02dd78479d52470b3c3a797408b20cc5a11e067e94b8
    THIS IS NOT YOUR TOKEN.  YOU HAVE TO SEARCH THE LOGS TO GET YOUR TOKEN
```

The token is the value behind `/?token=`. You need that for logging in.

**Note:** You can also run following command to get token directly: 
`docker exec -it jupyter bash -c 'jupyter notebook list' | cut -d'=' -f 2 | cut -d' ' -f 1`

Next, you can open the Jupyter Notebook at 
 https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/

 ### Linear regression

 Let's talk about regression analysis, a very popular topic in data science and statistics. It's all about trying to fit a curve or some sort of function, to a set of observations and then using that function to predict new values that you haven't seen yet. That's all there is to linear regression!

So, linear regression is fitting a straight line to a set of observations. For example, let's say that I have a bunch of people that I measured and the two features that I measured of these people are their weight and their height:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/1.png)

I'm showing the weight on the x-axis and the height on the y-axis, and I can plot all these data points, as in the people's weight versus their height, and I can say, "Hmm, that looks like a linear relationship, doesn't it? Maybe I can fit a straight line to it and use that to predict new values", and that's what linear regression does. In this example, I end up with a slope of 0.6 and a y-intercept of 130.2 which define a straight line (the equation of a straight line is y=mx+b, where m is the slope and b is the y-intercept). Given a slope and a y-intercept, that fits the data that I have best, I can use that line to predict new values.

You can see that the weights that I observed only went up to people that weighed 100 kilograms. What if I had someone who weighed 120 kilograms? Well, I could use that line to then figure out where would the height be for someone with 120 kilograms based on this previous data.

I don't know why they call it regression. Regression kind of implies that you're doing something backwards. I guess you can think of it in terms of you're creating a line to predict new values based on observations you made in the past, backwards in time, but it seems like a little bit of a stretch. It's just a confusing term quite honestly, and one way that we kind of obscure what we do with very simple concepts using very fancy terminology. All it is, is fitting a straight line to a set of data points.

### The ordinary least squares technique

How does linear regression work? Well internally, it uses a technique called ordinary least squares; it's also known as, OLS. You might see that term tossed around as well. The way it works is it tries to minimize the squared error between each point and the line, where the error is just the distance between each point and the line that you have.

So, we sum up all the squares of those errors, which sounds a lot like when we computed variance, right, except that instead of relative to the mean, it's relative to the line that we're defining. We can measure the variance of the data points from that line, and by minimizing that variance, we can find the line that fits it the best:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/2.png)

Now you'll never have to actually do this yourself the hard way, but if you did have to for some reason, or if you're just curious about what happens under the hood, I'll now describe the overall algorithm for you and how you would actually go about computing the slope and y-intercept yourself the hard way if you need to one day. It's really not that complicated.

Remember the slope-intercept equation of a line? It is `y=mx+c`. The slope just turns out to be the correlation between the two variables times the standard deviation in Y divided by the standard deviation in X. It might seem a little bit weird that standard deviation just kind of creeps into the math naturally there, but remember correlation had standard deviation baked into it as well, so it's not too surprising that you have to reintroduce that term.

The intercept can then be computed as the mean of the `Y` minus the slope times the mean of `X`. Again,even though that's really not that difficult, Python will do it all for you, but the point is that these aren't complicated things to run. They can actually be done very efficiently.

Remember that least squares minimize the sum of squared errors from each point to the line. Another way of thinking about linear regression is that you're defining a line that represents the maximum likelihood of an observation line there; that is, the maximum probability of the y value being something for a given x value.

People sometimes call linear regression maximum likelihood estimation, and it's just another example of people giving a fancy name to something that's very simple, so if you hear someone talk about maximum likelihood estimation, they're really talking about regression. They're just trying to sound really smart. But now you know that term too, so you too can sound smart.

### The gradient descent technique

There is more than one way to do linear regression. We've talked about ordinary least squares as being a simple way of fitting a line to a set of data, but there are other techniques as well, gradient descent being one of them, and it works best in three-dimensional data. So, it tries to follow the contours of the data for you. It's very fancy and obviously a little bit more computationally expensive, but Python does make it easy for you to try it out if you want to compare it to ordinary least squares.

**Note:**

Using the gradient descent technique can make sense when dealing with 3D data.

Usually though, least squares is a perfectly good choice for doing linear regression, and it's always a legitimate thing to do, but if you do run into gradient descent, you will know that that is just an alternate way of doing linear regression, and it's usually seen in higher dimensional data.

### The co-efficient of determination or r-squared


So how do I know how good my regression is? How well does my line fit my data? That's where r-squared comes in, and r-squared is also known as the coefficient of determination. Again, someone trying to sound smart might call it that, but usually it's called r-squared.

It is the fraction of the total variation in Y that is captured by your models. So how well does your line follow that variation that's happening? Are we getting an equal amount of variance on either side of your line or not? That's what r-squared is measuring.

Computing r-squared
To actually compute the value, take 1 minus the sum of the squared errors over the sum of the squared variations from the mean:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/3.png)

So, it's not very difficult to compute, but again, Python will give you functions that will just compute that for you, so you'll never have to actually do that math yourself.

### Interpreting r-squared

For r-squared, you will get a value that ranges from 0 to 1. Now 0 means your fit is terrible. It doesn't capture any of the variance in your data. While 1 is a perfect fit, where all of the variance in your data gets captured by this line, and all of the variance you see on either side of your line should be the same in that case. So 0 is bad, and 1 is good. That's all you really need to know. Something in between is something in between. A low r-squared value means it's a poor fit, a high r-squared value means it's a good fit.

As you'll see in the coming sections, there's more than one way to do regression. Linear regression is one of them. It's a very simple technique, but there are other techniques as well, and you can use r-squared as a quantitative measure of how good a given regression is to a set of data points, and then use that to choose the model that best fits your data.

### Computing linear regression and r-squared using Python


Let's now play with linear regression and actually compute some linear regression and r-squared. We can start by creating a little bit of Python code here that generates some random-ish data that is in fact linearly correlated.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `LinearRegression.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/LinearRegression.ipynb

In this example I'm going to fake some data about page rendering speeds and how much people purchase, just like a previous example. We're going to fabricate a linear relationship between the amount of time it takes for a website to load and the amount of money people spend on that website:

```
%matplotlib inline
import numpy as np
from pylab import *
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds + np.random.normal(0, 0.1, 1000)) * 3
scatter(pageSpeeds, purchaseAmount) 
```

All I've done here is I've made a random, a normal distribution of page speeds centered around 3 seconds with a standard deviation of 1 second. I've made the purchase amount a linear function of that. So, I'm making it 100 minus the page speeds plus some normal random distribution around it, times 3. And if we scatter that, we can see that the data ends up looking like this:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/4.png)

You can see just by eyeballing it that there's definitely a linear relationship going on there, and that's because we did hardcode a real linear relationship in our source data.


Now let's see if we can tease that out and find the best fit line using ordinary least squares. We talked about how to do ordinary least squares and linear regression, but you don't have to do any of that math yourself because the SciPy package has a stats package that you can import:

```
from scipy import stats

slope, intercept, r_value, p_value, std_err =     
stats.linregress(pageSpeeds, purchaseAmount) 
```

You can import stats from scipy, and then you can just call stats.linregress() on your two features. So, we have a list of page speeds (pageSpeeds) and a corresponding list of purchase amounts (purchaseAmount). The linregress() function will give us back a bunch of stuff, including the slope, the intercept, which is what I need to define my best fit line. It also gives us the r_value, from which we can get r-squared to measure the quality of that fit, and a couple of things that we'll talk about later on. For now, we just need slope, intercept, and r_value, so let's go ahead and run these. We'll begin by finding the linear regression best fit:

```
r_value ** 2
```

This is what your output should look like:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/5-0.png)

Now the r-squared value of the line that we got back is 0.99, that's almost 1.0. That means we have a really good fit, which isn't too surprising because we made sure there was a real linear relationship between this data. Even though there is some variance around that line, our line captures that variance. We have roughly the same amount of variance on either side of the line, which is a good thing. It tells us that we do have a linear relationship and our model is a good fit for the data that we have.

Let's plot that line:

```
import matplotlib.pyplot as plt
def predict(x):
return slope * x + intercept
fitLine = predict(pageSpeeds)
plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c='r')
plt.show()
```

The following is the output to the preceding code:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/5.png)

This little bit of code will create a function to draw the best fit line alongside the data. There's a little bit more Matplotlib magic going on here. We're going to make a fitLine list and we're going to use the predict() function we wrote to take the pageSpeeds,which is our x-axis, and create the Y function from that. So instead of taking the observations for amount spent, we're going to find the predicted ones just using the slope times x plus the intercept that we got back from the linregress() call above. Essentially here, we're going to do a scatter plot like we did before to show the raw data points, which are the observations.

Then we're also going to call plot on that same pyplot instance using our fitLine that we created using the line equation that we got back, and show them all both together. When we do that, it looks like the following graph:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/6.png)

You can see that our line is in fact a great fit for our data! It goes right smack down the middle, and all you need to predict new values is this predict function. Given a new previously unseen page speed, we could predict the amount spent just using the slope times the page speed plus the intercept. That's all there is to it, and I think it's great!

### Activity for linear regression

Time now to get your hands dirty. Try increasing the random variation in the test data and see if that has any impact. Remember, the r-squared is a measure of the fit, of how much do we capture the variance, so the amount of variance, well... why don't you see if it actually makes a difference or not.

That's linear regression, a pretty simple concept. All we're doing is fitting a straight line to set of observations, and then we can use that line to make predictions of new values. That's all there is to it. But why limit yourself to a line? There's other types of regression we can do that are more complex. We'll explore these next.

### Polynomial regression

We've talked about linear regression where we fit a straight line to a set of observations. Polynomial regression is our next topic, and that's using higher order polynomials to fit your data. So, sometimes your data might not really be appropriate for a straight line. That's where polynomial regression comes in.

Polynomial regression is a more general case of regression. So why limit yourself to a straight line? Maybe your data doesn't actually have a linear relationship, or maybe there's some sort of a curve to it, right? That happens pretty frequently.


Not all relationships are linear, but the linear regression is just one example of a whole class of regressions that we can do. If you remember the linear regression line that we ended up with was of the form y = mx + b, where we got back the values m and b from our linear regression analysis from ordinary least squares, or whatever method you choose. Now this is just a first order or a first-degree polynomial. The order or the degree is the power of x that you see. So that's the first-order polynomial.

Now if we wanted, we could also use a second-order polynomial, which would look like y = ax^2 + bx + c. If we were doing a regression using a second-order polynomial, we would get back values for a, b, and c. Or we could do a third-order polynomial that has the form ax^3 + bx^2 + cx + d. The higher the orders get, the more complex the curves you can represent. So, the more powers of x you have blended together, the more complicated shapes and relationships you can get.

But more degrees aren't always better. Usually there's some natural relationship in your data that isn't really all that complicated, and if you find yourself throwing very large degrees at fitting your data, you might be overfitting!

**Beware of overfitting!**

- Don't use more degrees than you need
- Visualize your data first to see how complex of a curve there might really be
- Visualize the fit and check if your curve going out of its way to accommodate outliers
- A high r-squared simply means your curve fits your training data well; it may or may not be good predictor

If you have data that's kind of all over the place and has a lot of variance, you can go crazy and create a line that just like goes up and down to try to fit that data as closely as it can, but in fact that doesn't represent the intrinsic relationship of that data. It doesn't do a good job of predicting new values.

So always start by just visualizing your data and think about how complicated does the curve really needs to be. Now you can use r-squared to measure how good your fit is, but remember, that's just measuring how well this curve fits your training data—that is, the data that you're using to actually make your predictions based off of. It doesn't measure your ability to predict accurately going forward.

Later, we'll talk about some techniques for preventing overfitting called train/test, but for now you're just going to have to eyeball it to make sure that you're not overfitting and throwing more degrees at a function than you need to. This will make more sense when we explore an example, so let's do that next.

### Implementing polynomial regression using NumPy

Fortunately, NumPy has a polyfit function that makes it super easy to play with this and experiment with different results, so let's go take a look. Time for fun with polynomial regression. I really do think it's fun, by the way. It's kind of cool seeing all that high school math actually coming into some practical application. Go ahead and open the `PolynomialRegression.ipynb` and let's have some fun.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `PolynomialRegression.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/PolynomialRegression.ipynb

Let's create a new relationship between our page speeds, and our purchase amount fake data, and this time we're going to create a more complex relationship that's not linear. We're going to take the page speed and make it some function of the division of page speed for the purchase amount:

```
%matplotlib inline
from pylab import *
np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
scatter(pageSpeeds, purchaseAmount)
```

If we do a scatter plot, we end up with the following

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/7.png)

By the way, if you're wondering what the `np.random.seed` line does, it creates a random seed value, and it means that when we do subsequent random operations they will be deterministic. By doing this we can make sure that, every time we run this bit of code, we end up with the same exact results. That's going to be important later on because I'm going to suggest that you come back and actually try different fits to this data to compare the fits that you get. So, it's important that you're starting with the same initial set of points.

You can see that that's not really a linear relationship. We could try to fit a line to it and it would be okay for a lot of the data, maybe down at the right side of the graph, but not so much towards the left. We really have more of an exponential curve.

Now it just happens that NumPy has a polyfit() function that allows you to fit any degree polynomial you want to this data. So, for example, we could say our x-axis is an array of the page speeds (pageSpeeds) that we have, and our y-axis is an array of the purchase amounts (purchaseAmount) that we have. We can then just call np.polyfit(x, y, 4), meaning that we want a fourth degree polynomial fit to this data.

```
x = np.array(pageSpeeds)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x, y, 4))
```
Let's go ahead and run that. It runs pretty quickly, and we can then plot that. So, we're going to create a little graph here that plots our scatter plot of original points versus our predicted points.

```
import matplotlib.pyplot as plt

xp = np.linspace(0, 7, 100)
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
plt.show()
```

The output looks like the following graph:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/8.png)

At this point, it looks like a reasonably good fit. What you want to ask yourself though is, "Am I overfitting? Does my curve look like it's actually going out of its way to accommodate outliers?" I find that that's not really happening. I don't really see a whole lot of craziness going on.

If I had a really high order polynomial, it might swoop up at the top to catch that one outlier and then swoop downwards to catch the outliers there, and get a little bit more stable through where we have a lot of density, and maybe then it could potentially go all over the place trying to fit the last set of outliers at the end. If you see that sort of nonsense, you know you have too many orders, too many degrees in your polynomial, and you should probably bring it back down because, although it fits the data that you observed, it's not going to be useful for predicting data you haven't seen.

Imagine I have some curve that swoops way up and then back down again to fit outliers. My prediction for something in between there isn't going to be accurate. The curve really should be in the middle. Later in this book we'll talk about the main ways of detecting such overfitting, but for now, please just observe it and know we'll go deeper later.

### Computing the r-squared error

Now we can measure the r-squared error. By taking the y and the predicted values (p4(x)) in the r2_score() function that we have in sklearn.metrics, we can compute that.

```
from sklearn.metrics import r2_score
r2 = r2_score(y, p4(x))

print r2
```

The output is as follows:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/9-0.png)

Our code compares a set of observations to a set of predictions and computes r-squared for you, and with just one line of code! Our r-squared for this turns out to be 0.829, which isn't too bad. Remember, zero is bad, one is good. 0.82 is to pretty close to one, not perfect, and intuitively, that makes sense. You can see that our line is pretty good in the middle section of the data, but not so good out at the extreme left and not so good down at the extreme right. So, 0.82 sounds about right.

### Activity for polynomial regression

I recommend that you get down and dirty with this stuff. Try different orders of polynomials. Go back up to where we ran the polyfit() function and try different values there besides 4. You can use 1, and that would go back to a linear regression, or you could try some really high amount like 8, and maybe you'll start to see overfitting. So see what effect that has. You're going to want to change that. For example, let's go to a third-degree polynomial.

```
x = np.array(pageSpeeds)
y = np.array(purchaseAmount)

p4 = np.poly1d(np.polyfit(x, y, 3))  
```

Just keep hitting run to go through each step and you can see the it's effect as...

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/9.png)

Our third-degree polynomial is definitely not as good a fit as the fourth-degree polynomial. If you actually measure the r-squared error, it would actually turn out worse, quantitatively; but if I go too high, you might start to see overfitting. So just have some fun with that, play around different values, and get a sense of what different orders of polynomials do to your regression. Go get your hands dirty and try to learn something.

So that's polynomial regression. Again, you need to make sure that you don't put more degrees at the problem than you need to. Use just the right amount to find what looks like an intuitive fit to your data. Too many can lead to overfitting, while too few can lead to a poor fit... so you can use both your eyeballs for now, and the r-squared metric, to figure out what the right number of degrees are for your data. Let's move on.

### Multivariate regression and predicting car prices

What happens then, if we're trying to predict some value that is based on more than one other attribute? Let's say that the height of people not only depends on their weight, but also on their genetics or some other things that might factor into it. Well, that's where multivariate analysis comes in. You can actually build regression models that take more than one factor into account at once. It's actually pretty easy to do with Python.

Let's talk about multivariate regression, which is a little bit more complicated. The idea of multivariate regression is this: what if there's more than one factor that influences the thing you're trying to predict?

In our previous examples, we looked at linear regression. We talked about predicting people's heights based on their weight, for example. We assumed that the weight was the only thing that influenced their height, but maybe there are other factors too. We also looked at the effect of page speed on purchase amounts. Maybe there's more that influences purchase amounts than just page speed, and we want to find how these different factors all combine together to influence that value. So that's where multivariate regression comes in.

The example we're going to look at now is as follows. Let's say that you're trying to predict the price that a car will sell for. It might be based on many different features of that car, such as the body style, the brand, the mileage; who knows, even on how good the tires are. Some of those features are going to be more important than others toward predicting the price of a car, but you want to take into account all of them at once.

So our way forwards here is still going to use the least-squares approach to fit a model to your set of observations. The difference is that we're going to have a bunch of coefficients for each different feature that you have.

So, for example, the price model that we end up with might be a linear relationship of alpha, some constant, kind of like your y-intercept was, plus some coefficient of the mileage, plus some coefficient of the age, plus some coefficient of how many doors it has:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/steps/22/1.png)

Once you end up with those coefficients, from least squares analysis, we can use that information to figure out, well, how important are each of these features to my model. So, if I end up with a very small coefficient for something like the number of doors, that implies that the number of doors isn't that important, and maybe I should just remove it from my model entirely to keep it simpler.

You always want to do the simplest thing that works in data science. Don't over complicate things, because it's usually the simple models that work the best. If you can find just the right amount of complexity, but no more, that's usually the right model to go with. Anyway, those coefficients give you a way of actually, "Hey some of these things are more important than others. Maybe I can discard some of these factors."

Now we can still measure the quality of a fit with multivariate regression using r-squared. It works the same way, although one thing you need to assume when you're doing multivariate regression is that the factors themselves are not dependent on each other... and that's not always true. So sometimes you need to keep that little caveat in the back of your head. For example, in this model we're going to assume that mileage and age of the car are not related; but in fact, they're probably pretty tightly related! This is a limitation of this technique, and it might not be capturing an effect at all.

### Multivariate regression using Python

The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `MultivariateRegression.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/MultivariateRegression.ipynb

Fortunately there's a statsmodel package available for Python that makes doing multivariate regression pretty easy. Let's just dive in and see how it works. Let's do some multivariate regression using Python. We're going to use some real data here about car values from the Kelley Blue Book.

```
import pandas as pd
df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')
```

We're going to introduce a new package here called pandas, which lets us deal with tabular data really easily. It lets us read in tables of data and rearrange them, and modify them, and slice them and dice them in different ways. We're going to be using that a lot going forward.

We're going to import pandas as pd, and pd has a read_Excel() function that we can use to go ahead and read a Microsoft Excel spreadsheet from the Web through HTTP. So, pretty awesome capabilities of pandas there.


I've gone ahead and hosted that file for you on my own domain, and if we run that, it will load it into what's called a DataFrame object that we're referring to as df. Now I can call head() on this DataFrame to just show the first few lines of it:

```
df.head()
```

The following is the output for the preceding code:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/10.png)

The actual dataset is much larger. This is just the first few samples. So, this is real data of mileage, make, model, trim, type, doors, cruise, sound and leather.

OK, now we're going to use pandas to split that up into the features that we care about. We're going to create a model that tries to predict the price just based on the mileage, the model, and the number of doors, and nothing else.

```
import statsmodels.api as sm

df['Model_ord'] = pd.Categorical(df.Model).codes
X = df[['Mileage', 'Model_ord', 'Doors']]
y = df[['Price']]

X1 = sm.add_constant(X)
est = sm.OLS(y, X1).fit()

est.summary() 
```

Now the problem that I run into is that the model is a text, like Century for Buick, and as you recall, everything needs to be a number when I'm doing this sort of analysis. In the code, I use this Categorical() function in pandas to actually convert the set of model names that it sees in the DataFrame into a set of numbers; that is, a set of codes. I'm going to say my input for this model on the x-axis is mileage (Mileage), model converted to an ordinal value (Model_ord), and the number of doors (Doors). What I'm trying to predict on the y-axis is the price (Price).


The next two lines of the code just create a model that I'm calling est that uses ordinary least squares, OLS, and fits that using the columns that I give it, Mileage, Model_ord, and Doors. Then I can use the summary call to print out what my model looks like:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/11.png)

You can see here that the r-squared is pretty low. It's not that good of a model, really, but we can get some insight into what the various errors are, and interestingly, the lowest standard error is associated with the mileage.

Now I have said before that the coefficient is a way of determining which items matter, and that's only true though if your input data is normalized. That is, if everything's on the same scale of 0 to 1. If it's not, then these coefficients are kind of compensating for the scale of the data that it's seeing. If you're not dealing with normalized data, as in this case, it's more useful to look at the standard errors. In this case, we can see that the mileage is actually the biggest factor of this particular model. Could we have figured that out earlier? Well, we could have just done a little bit of slicing and dicing to figure out that the number of doors doesn't actually influence the price much at all. Let's run the following little line:

```
y.groupby(df.Doors).mean()
```

A little bit of pandas syntax there. It's pretty cool that you can do it in Python in one line of code! That will print out a new DataFrame that shows the mean price for the given number of doors:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-04/12.png)

I can see the average two-door car sells for actually more than the average four-door car. If anything there's a negative correlation between number of doors and price, which is a little bit surprising. This is a small dataset, though, so we can't read a whole lot of meaning into it of course.

### Activity for multivariate regression

As an activity, please mess around with the fake input data where you want. You can download the data and mess around with the spreadsheet. Read it from your local hard drive instead of from HTTP, and see what kind of differences you can have. Maybe you can fabricate a dataset that has a different behavior and has a better model that fits it. Maybe you can make a wiser choice of features to base your model off of. So, feel free to mess around with that and let's move on.

There you have it: multivariate analysis and an example of it running. Just as important as the concept of multivariate analysis, which we explored, was some of the stuff that we did in that Python notebook. So, you might want to go back there and study exactly what's going on.

We introduced pandas and the way to work with pandas and DataFrame objects. pandas a very powerful tool. We'll use it more in future sections, but make sure you're starting to take notice of these things because these are going to be important techniques in your Python skills for managing large amounts of data and organizing your data.

### Multi-level models


It makes sense now to talk about multi-level models. This is definitely an advanced topic, and I'm not going to get into a whole lot of detail here. My objective right now is to introduce the concept of multi-level models to you, and let you understand some of the challenges and how to think about them when you're putting them together. That's it.

The concept of multi-level models is that some effects happen at various levels in the hierarchy. For example, your health. Your health might depend on how healthy your individual cells are, and those cells might be a function of how healthy the organs that they're inside are, and the health of your organs might depend on the health of you as a whole. Your health might depend in part on your family's health and the environment your family gives you. And your family's health in turn might depend on some factors of the city that you live in, how much crime is there, how much stress is there, how much pollution is there. And even beyond that, it might depend on factors in the entire world that we live in. Maybe just the state of medical technology in the world is a factor, right?

Another example: your wealth. How much money do you make? Well, that's a factor of your individual hard work, but it's also a factor of the work that your parents did, how much money were they able to invest into your education and the environment that you grew up in, and in turn, how about your grandparents? What sort of environment were they able to create and what sort of education were they able to offer for your parents, which in turn influenced the resources they have available for your own education and upbringing.

These are all examples of multi-level models where there is a hierarchy of effects that influence each other at larger and larger scales. Now the challenge of multi-level models is to try to figure out, "Well, how do I model these interdependencies? How do I model all these different effects and how they affect each other?"

The challenge here is to identify the factors in each level that actually affect the thing you're trying to predict. If I'm trying to predict overall SAT scores, for example, I know that depends in part on the individual child that's taking the test, but what is it about the child that matters? Well, it might be the genetics, it might be their individual health, the individual brain size that they have. You can think of any number of factors that affect the individual that might affect their SAT score. And then if you go up another level, look at their home environment, look at their family. What is it about their families that might affect their SAT scores? How much education were they able to offer? Are the parents able to actually tutor the children in the topics that are on the SAT? These are all factors at that second level that might be important. What about their neighborhood? The crime rate of the neighborhood might be important. The facilities they have for teenagers and keeping them off the streets, things like that.


The point is you want to keep looking at these higher levels, but at each level identify the factors that impact the thing you're trying to predict. I can keep going up to the quality of the teachers in their school, the funding of the school district, the education policies at the state level. You can see there are different factors at different levels that all feed into this thing you're trying to predict, and some of these factors might exist at more than one level. Crime rate, for example, exists at the local and state levels. You need to figure out how those all interplay with each other as well when you're doing multi-level modeling.

As you can imagine, this gets very hard and very complicated very quickly. It is really way beyond the scope of this book, or any introductory book in data science. This is hard stuff. There are entire thick books about it, you could do an entire book about it that would be a very advanced topic.


So why am I even mentioning multi-level models? It is because I've seen it mentioned on job descriptions, in a couple of cases, as something that they want you to know about in a couple of cases. I've never had to use it in practice, but I think the important thing from the standpoint of getting a career in data science is that you at least are familiar with the concept, and you know what it means and some of the challenges involved in creating a multi-level model. I hope I've given you those concepts. With that, we can move on to the next section.

There you have the concepts of multi-level models. It's a very advanced topic, but you need to understand what the concept is, at least, and the concept itself is pretty simple. You just are looking at the effects at different levels, different hierarchies when you're trying to make a prediction. So maybe there are different layers of effects that have impacts on each other, and those different layers might have factors that interrelate with each other as well. Multi-level modeling tries to take account of all those different hierarchies and factors and how they interplay with each other. Rest assured that's all you need to know for now.
