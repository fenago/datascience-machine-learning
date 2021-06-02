<img align="right" src="../images/logo-small.png">


Lab : Statistics and Probability Refresher, and Python Practice - Part 2
-------------------------------------


We'll be covering the following topics in this lab:

- Types of data distributions and how to plot them
- Understanding percentiles and moments

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/datascience-machine-learning` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab_


 ### Types of data distributions

Let's look at some real examples of probability distribution functions and data distributions in general and wrap your head a little bit more around data distributions and how to visualize them and use them in Python.

Go ahead and open up the Distributions.ipynb from the book materials, and you can follow along with me here if you'd like.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `Distributions.ipynb` in the `work` folder.



### Uniform distribution

Let's start off with a really simple example: uniform distribution. A uniform distribution just means there's a flat constant probability of a value occurring within a given range.

```
import numpy as np 
Import matplotlib.pyplot as plt 
 
values = np.random.uniform(-10.0, 10.0, 100000) 
plt.hist(values, 50) 
plt.show() 
```

So we can create a uniform distribution by using the NumPy random.uniform function. The preceding code says, I want a uniformly distributed random set of values that ranges between -10 and 10, and I want 100000 of them. If I then create a histogram of those values, you can see it looks like the following.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-02-01/steps/2/1.png)

There's pretty much an equal chance of any given value or range of values occurring within that data. So, unlike the normal distribution, where we saw a concentration of values near the mean, a uniform distribution has equal probability across any given value within the range that you define.

So what would the probability distribution function of this look like? Well, I'd expect to see basically nothing outside of the range of `-10` or beyond `10`. But when I'm between -10 and 10, I would see a flat line because there's a constant probability of any one of those ranges of values occurring. So in a uniform distribution you would see a flat line on the probability distribution function because there is basically a constant probability. Every value, every range of values has an equal chance of appearing as any other value.

### Normal or Gaussian distribution

Now we've seen normal, also known as Gaussian, distribution functions already in this course. You can actually visualize those in Python. There is a function called pdf (probability density function) in the scipy.stats.norm package function.

So, let's look at the following example:

```
from scipy.stats import norm 
import matplotlib.pyplot as plt 
 
x = np.arange(-3, 3, 0.001) 
plt.plot(x, norm.pdf(x))
```

In the preceding example, we're creating a list of x values for plotting that range between -3 and 3 with an increment of 0.001 in between them by using the arange function. So those are the x values on the graph and we're going to plot the x-axis with using those values. The y-axis is going to be the normal function, norm.pdf, that the probability density function for a normal distribution, on those x values. We end up with the following output:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-02-01/steps/2/2.png)

The pdf function with a normal distribution looks just like it did in our previous section, that is, a normal distribution for the given numbers that we provided, where 0 represents the mean, and the numbers -3, -2, -1, 1, 2, and 3 are standard deviations.

Now, we will generate random numbers with a normal distribution. We've done this a few times already; consider this a refresher. Refer to the following block of code:

```
import numpy as np 
import matplotlib.pyplot as plt 
 
mu = 5.0 
sigma = 2.0 
values = np.random.normal(mu, sigma, 10000) 
plt.hist(values, 50) 
plt.show() 
```

In the above code, we use the random.normal function of the NumPy package, and the first parameter mu, represents the mean that you want to center the data around. sigma is the standard deviation of that data, which is basically the spread of it. Then, we specify the number of data points that we want using a normal probability distribution function, which is 10000 here. So that's a way to use a probability distribution function, in this case the normal distribution function, to generate a set of random data. We can then plot that, using a histogram broken into 50 buckets and show it. The following output is what we end up with:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-02-01/steps/2/3.png)

It does look more or less like a normal distribution, but since there is a random element, it's not going to be a perfect curve. We're talking about probabilities; there are some odds of things not quite being what they should be.

### The exponential probability distribution or Power law

Another distribution function you see pretty often is the exponential probability distribution function, where things fall off in an exponential manner.

When you talk about an exponential fall off, you expect to see a curve, where it's very likely for something to happen, near zero, but then, as you get farther away from it, it drops off very quickly. There's a lot of things in nature that behave in this manner.

To do that in Python, just like we had a function in scipy.stats for norm.pdf, we also have an expon.pdf, or an exponential probability distribution function to do that in Python, we can do the same syntax that we did for the normal distribution with an exponential distribution here as shown in the following code block:

```
from scipy.stats import expon 
import matplotlib.pyplot as plt 
 
x = np.arange(0, 10, 0.001) 
plt.plot(x, expon.pdf(x)) 
```

So again, in the above code, we just create our x values using the NumPy arange function to create a bunch of values between 0 and 10 with a step size of `0.001`. Then, we plot those x values against the y-axis, which is defined as the function `expon.pdf(x)`. The output looks like an exponential fall off. As shown in the following screenshot:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-02-01/steps/2/4.png)


### Binomial probability mass function

We can also visualize probability mass functions. This is called the binomial probability mass function. Again, we are going to use the same syntax as before, as shown in the following code:

```
from scipy.stats import expon 
import matplotlib.pyplot as plt 
 
x = np.arange(0, 10, 0.001) 
plt.plot(x, expon.pdf(x)) 
```

So instead of expon or norm, we just use binom. A reminder: The probability mass function deals with discrete data. We have been all along, really, it's just how you think about it.

Coming back to our code, we're creating some discrete x values between 0 and 10 at a spacing of 0.01, and we're saying I want to plot a binomial probability mass function using that data. With the binom.pmf function, I can actually specify the shape of that data using two shape parameters, n and p. In this case, they're 10 and 0.5 respectively. output is shown on the following graph:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-02-01/steps/2/5.png)

If you want to go and play around with different values to see what effects it has, that's a good way to get an intuitive sense of how those shape parameters work on the probability mass function.


Let's say I have a website, and on average I get 500 visitors per day. I can use the Poisson probability mass function to estimate the probability of seeing some other value on a specific day. For example, with my average of 500 visitors per day, what's the odds of seeing 550 visitors on a given day? That's what a Poisson probability mass function can give you take a look at the following code:

```
from scipy.stats import poisson 
import matplotlib.pyplot as plt 
 
mu = 500 
x = np.arange(400, 600, 0.5) 
plt.plot(x, poisson.pmf(x, mu)) 
```

In this code example, I'm saying my average is 500 mu. I'm going to set up some x values to look at between 400 and 600 with a spacing of 0.5. I'm going to plot that using the poisson.pmf function. I can use that graph to look up the odds of getting any specific value that's not `500`, assuming a normal distribution:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-02-01/steps/2/6.png)

The odds of seeing 550 visitors on a given day, it turns out, comes out to about 0.002 or 0.2% probability. Very interesting.

Alright, so those are some common data distributions you might run into in the real world.

**Note:**

Remember we used a probability distribution function with continuous data, but when we're dealing with discrete data, we use a probability mass function.

So that's probability density functions, and probability mass functions. Basically, a way to visualize and measure the actual chance of a given range of values occurring in a dataset. Very important information and a very important thing to understand. We're going to keep using that concept over and over again. Alright, let's move on.

### Percentiles and moments

Next, we'll talk about percentiles and moments. You hear about percentiles in the news all the time. People that are in the top 1% of income: that's an example of percentile. We'll explain that and have some examples. Then, we'll talk about moments, a very fancy mathematical concept, but it turns out it's very simple to understand conceptually. Let's dive in and talk about percentiles and moments, a couple of a pretty basic concepts in statistics, but again, we're working our way up to the hard stuff, so bear with me as we go through some of this review.



### Percentiles

A common example you see talked about a lot, is income distribution. When we talk about the 99th percentile, or the one-percenters, imagine that you were to take all the incomes of everybody in the country, in this case the United States, and sort them by income. The 99th percentile will be the income amount at which 99% of the rest of the country was making less than that amount. It's a very easy way to comprehend it.


### Computing percentiles in Python

Let's look at some more examples of percentiles using Python and kind of get our hands on it and conceptualize this a little bit more. Go ahead and open the Percentiles.ipynb file if you'd like to follow along.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for notebooks. Open and run `Percentiles.ipynb` in the `work` folder.




Let's start off by generating some randomly distributed normal data, or normally distributed random data, rather, refer to the following code block:

```
%matplotlib inline 
import numpy as np 
import matplotlib.pyplot as plt 
 
vals = np.random.normal(0, 0.5, 10000) 
 
plt.hist(vals, 50) 
plt.show() 
```

In this example, what we're going to do is generate some data centered around zero, that is with a mean of zero, with a standard deviation of 0.5, and I'm going to make 10000 data points with that distribution. Then, we're going to plot a histogram and see what we come up with.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-02-01/steps/9/3.png)

The generated histogram looks very much like a normal distribution, but because there is a random component we have a little outlier near the deviation of -2 in this example here. Things are tipped a little bit at the mean, a little bit of random variation there to make things interesting.

NumPy provides a very handy percentile function that will compute the percentile values of this distribution for you. So, we created our vals list of data using np.random.normal, and I can just call the np.percentile function to figure out the 50th percentile value in using the following code:

```
np.percentile(vals, 50) 
```

The following is the output of the preceding code:

```
0.0053397035195310248
```

The output turns out to be 0.005. So remember, the 50th percentile is just another name for the median, and it turns out the median is very close to zero in this data. You can see in the graph that we're tipped a little bit to the right, so that's not too surprising.

I want to compute the 90th percentile, which gives me the point at which 90% of the data is less than that value. We can easily do that with the following code:

```
np.percentile(vals, 90)
```

Here is the output of that code:

```
Out[4]: 0.64099069837340827 
```

The 90th percentile of this data turns out to be 0.64, so it's around here, and basically, at that point less than 90% of the data is less than that value. I can believe that. 10% of the data is greater than 0.64, 90% of it, less than 0.65.

Let's compute the 20th percentile value, that would give me the point at which 20% of the values are less than that number that I come up with. Again, we just need a very simple alteration to the code:

```
np.percentile(vals, 20) 
```

This gives the following output:

```
Out[5]:-0.41810340026619164 
```

The 20th percentile point works out to be -0.4, roughly, and again I believe that. It's saying that 20% of the data lies to the left of -0.4, and conversely, 80% is greater.

If you want to get a feel as to where those breaking points are in a dataset, the percentile function is an easy way to compute them. If this were a dataset representing income distribution, we could just call np.percentile(vals, 99) and figure out what the 99th percentile is. You could figure out who those one-percenters people keep talking about really are, and if you're one of them.


### Moments

Basically, moments are ways to measure the shape of a data distribution, of a probability density function, or of anything, really. Mathematically, we've got some really fancy notation to define them:


### Computing moments in Python

Let's play around in Python and actually compute these moments and see how you do that. To play around with this, go ahead and open up the `Moments.ipynb`, and you can follow along with me here.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `Moments.ipynb` in the `work` folder.



Let's again create that same normal distribution of random data. Again, we're going to make it centered around zero, with a 0.5 standard deviation and 10,000 data points, and plot that out:

```
import numpy as np 
import matplotlib.pyplot as plt 
 
vals = np.random.normal(0, 0.5, 10000) 
 
plt.hist(vals, 50) 
plt.show()
```

So again, we get a randomly generated set of data with a normal distribution around zero.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-02-01/steps/9/7.png)

Now, we find the mean and variance. We've done this before; NumPy just gives you a mean and var function to compute that. So, we just call np.mean to find the first moment, which is just a fancy word for the mean, as shown in the following code:

```
np.mean(vals)
```

This gives the following output in our example:

```
Out [2]:-0.0012769999428169742
```

The output turns out to be very close to zero, just like we would expect for normally distributed data centered around zero. So, the world makes sense so far.

Now we find the second moment, which is just another name for variance. We can do that with the following code, as we've seen before:

```
np.var(vals)
```

Providing the following output:

```
Out[3]:0.25221246428323563
```

That output turns out to be about 0.25, and again, that works out with a nice sanity check. Remember that standard deviation is the square root of variance. If you take the square root of 0.25, it comes out to 0.5, which is the standard deviation we specified while creating this data, so again, that checks out too.


The third moment is skew, and to do that we're going to need to use the SciPy package instead of NumPy. But that again is built into any scientific computing package like Enthought Canopy or Anaconda. Once we have SciPy, the function call is as simple as our earlier two:

```
import scipy.stats as sp
sp.skew(vals)
```

This displays the following output:

```
Out[4]: 0.020055795996111746
```

We can just call sp.skew on the vals list, and that will give us the skew value. Since this is centered around zero, it should be almost a zero skew. It turns out that with random variation it does skew a little bit left, and actually that does jive with the shape that we're seeing in the graph. It looks like we did kind of pull it a little bit negative.

The fourth moment is kurtosis, which describes the shape of the tail. Again, for a normal distribution that should be about zero.SciPy provides us with another simple function call

```
sp.kurtosis(vals)
```
And here's the output:

```
Out [5]:0.059954502386585506
```

Indeed, it does turn out to be zero. Kurtosis reveals our data distribution in two linked ways: the shape of the tail, or the how sharp the peak If I just squish the tail down it kind of pushes up that peak to be pointier, and likewise, if I were to push down that distribution, you can imagine that's kind of spreading things out a little bit, making the tails a little bit fatter, and the peak of it a little bit lower. So that's what kurtosis means, and in this example, kurtosis is near zero because it is just a plain old normal distribution.

### Activity

If you want to play around with it, go ahead and, again, try to modify the distribution. Make it centered around something besides 0, and see if that actually changes anything. Should it? Well, it really shouldn't because these are all measures of the shape of the distribution, and it doesn't really say a whole lot about where that distribution is exactly. It's a measure of the shape. That's what the moments are all about. Go ahead and play around with that, try different center values, try different standard deviation values, and see what effect it has on these values, and it doesn't change at all. Of course, you'd expect things like the mean to change because you're changing the mean value, but variance, skew, maybe not. Play around, find out.
