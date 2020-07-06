### Introduction

In this scenario, we are going to go through a few concepts of statistics and probability, which might be a refresher for some of you. These concepts are important to go through if you want to be a data scientist. We will see examples to understand these concepts better. We will also look at how to implement those examples using actual Python code.

We'll be covering the following topics in this scenario:

- Types of data distributions and how to plot them
- Understanding percentiles and moments

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

 ### Types of data distributions

 Let's look at some real examples of probability distribution functions and data distributions in general and wrap your head a little bit more around data distributions and how to visualize them and use them in Python.

Go ahead and open up the Distributions.ipynb from the book materials, and you can follow along with me here if you'd like.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `Distributions.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/Distributions.ipynb

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

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/2/1.png)

There's pretty much an equal chance of any given value or range of values occurring within that data. So, unlike the normal distribution, where we saw a concentration of values near the mean, a uniform distribution has equal probability across any given value within the range that you define.

So what would the probability distribution function of this look like? Well, I'd expect to see basically nothing outside of the range of `-10` or beyond `10`. But when I'm between -10 and 10, I would see a flat line because there's a constant probability of any one of those ranges of values occurring. So in a uniform distribution you would see a flat line on the probability distribution function because there is basically a constant probability. Every value, every range of values has an equal chance of appearing as any other value.

### Normal or Gaussian distribution

Now we've seen normal, also known as Gaussian, distribution functions already in this book. You can actually visualize those in Python. There is a function called pdf (probability density function) in the scipy.stats.norm package function.

So, let's look at the following example:

```
from scipy.stats import norm 
import matplotlib.pyplot as plt 
 
x = np.arange(-3, 3, 0.001) 
plt.plot(x, norm.pdf(x))
```

In the preceding example, we're creating a list of x values for plotting that range between -3 and 3 with an increment of 0.001 in between them by using the arange function. So those are the x values on the graph and we're going to plot the x-axis with using those values. The y-axis is going to be the normal function, norm.pdf, that the probability density function for a normal distribution, on those x values. We end up with the following output:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/2/2.png)

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

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/2/3.png)

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

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/2/4.png)


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

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/2/5.png)

If you want to go and play around with different values to see what effects it has, that's a good way to get an intuitive sense of how those shape parameters work on the probability mass function.

Lastly, the other distribution function you might hear about is a Poisson probability mass function, and this has a very specific application. It looks a lot like a normal distribution, but it's a little bit different.

The idea here is, if you have some information about the average number of things that happen in a given time period, this probability mass function can give you a way to predict the odds of getting another value instead, on a given future day.

As an example, let's say I have a website, and on average I get 500 visitors per day. I can use the Poisson probability mass function to estimate the probability of seeing some other value on a specific day. For example, with my average of 500 visitors per day, what's the odds of seeing 550 visitors on a given day? That's what a Poisson probability mass function can give you take a look at the following code:

```
from scipy.stats import poisson 
import matplotlib.pyplot as plt 
 
mu = 500 
x = np.arange(400, 600, 0.5) 
plt.plot(x, poisson.pmf(x, mu)) 
```

In this code example, I'm saying my average is 500 mu. I'm going to set up some x values to look at between 400 and 600 with a spacing of 0.5. I'm going to plot that using the poisson.pmf function. I can use that graph to look up the odds of getting any specific value that's not `500`, assuming a normal distribution:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/2/6.png)

The odds of seeing 550 visitors on a given day, it turns out, comes out to about 0.002 or 0.2% probability. Very interesting.

Alright, so those are some common data distributions you might run into in the real world.

**Note:**

Remember we used a probability distribution function with continuous data, but when we're dealing with discrete data, we use a probability mass function.

So that's probability density functions, and probability mass functions. Basically, a way to visualize and measure the actual chance of a given range of values occurring in a dataset. Very important information and a very important thing to understand. We're going to keep using that concept over and over again. Alright, let's move on.

### Percentiles and moments

Next, we'll talk about percentiles and moments. You hear about percentiles in the news all the time. People that are in the top 1% of income: that's an example of percentile. We'll explain that and have some examples. Then, we'll talk about moments, a very fancy mathematical concept, but it turns out it's very simple to understand conceptually. Let's dive in and talk about percentiles and moments, a couple of a pretty basic concepts in statistics, but again, we're working our way up to the hard stuff, so bear with me as we go through some of this review.

### Percentiles

Let's see what percentiles mean. Basically, if you were to sort all of the data in a dataset, a given percentile is the point at which that percent of the data is less than the point you're at.

A common example you see talked about a lot, is income distribution. When we talk about the 99th percentile, or the one-percenters, imagine that you were to take all the incomes of everybody in the country, in this case the United States, and sort them by income. The 99th percentile will be the income amount at which 99% of the rest of the country was making less than that amount. It's a very easy way to comprehend it.

**Note:**

In a dataset, a percentile is the point at which x% of the values are less than the value at that point.

The following graph is an example for income distribution:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/9/1.png)

The preceding image shows an example of income distribution data. For example, at the 99th percentile we can say that 99% of the data points, which represent people in America, make less than $506,553 a year, and one percent make more than that. Conversely, if you're a one-percenter, you're making more than $506,553 a year. Congratulations! But if you're a more typical median person, the 50th percentile defines the point at which half of the people are making less and half are making more than you are, which is the definition of median. The 50th percentile is the same thing as median, and that would be at $42,327 given this dataset. So, if you're making $42,327 a year in the US, you are making exactly the median amount of income for the country.

You can see the problem of income distribution in the graph above. Things tend to be very concentrated toward the high end of the graph, which is a very big political problem right now in the country. We'll see what happens with that, but that's beyond the scope of this book. That's percentiles in a nutshell.

### Quartiles

Percentiles are also used in the context of talking about the quartiles in a distribution. Let's look at a normal distribution to understand this better.

Here's an example illustrating Percentile in normal distribution:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/9/2.png)

Looking at the normal distribution in the preceding image, we can talk about quartiles. Quartile 1 (Q1) and quartile 3 (Q3) in the middle are just the points that contain together 50% of the data, so 25% are on left side of the median and 25% are on the right side of the median.

The median in this example happens to be near the mean. For example, the interquartile range (IQR), when we talk about a distribution, is the area in the middle of the distribution that contains 50% of the values.

The topmost part of the image is an example of what we call a box-and-whisker diagram. Don't concern yourself yet about the stuff out on the edges of the box. That gets a little bit confusing, and we'll cover that later. Even though they are called quartile 1 (Q1) and quartile 3 (Q1), they don't really represent 25% of the data, but don't get hung up on that yet. Focus on the point that the quartiles in the middle represent 25% of the data distribution.

### Computing percentiles in Python

Let's look at some more examples of percentiles using Python and kind of get our hands on it and conceptualize this a little bit more. Go ahead and open the Percentiles.ipynb file if you'd like to follow along.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for notebooks. Open and run `Percentiles.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/Percentiles.ipynb


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

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/9/3.png)

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

Alright, now to get your hands dirty. I want you to play around with this data. This is an IPython Notebook for a reason, so you can mess with it and mess with the code, try different standard deviation values, see what effect it has on the shape of the data and where those percentiles end up lying, for example. Try using smaller dataset sizes and add a little bit more random variation in the thing. Just get comfortable with it, play around with it, and find you can actually do this stuff and write some real code that works.

### Moments

Next, let's talk about moments. Moments are a fancy mathematical phrase, but you don't actually need a math degree to understand it, though. Intuitively, it's a lot simpler than it sounds.

It's one of those examples where people in statistics and data science like to use big fancy terms to make themselves sound really smart, but the concepts are actually very easy to grasp, and that's the theme you're going to hear again and again in this book.

Basically, moments are ways to measure the shape of a data distribution, of a probability density function, or of anything, really. Mathematically, we've got some really fancy notation to define them:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/9/4.png)

If you do know calculus, it's actually not that complicated of a concept. We're taking the difference between each value from some value raised to the nth power, where n is the moment number and integrating across the entire function from negative infinity to infinity. But intuitively, it's a lot easier than calculus.

**Note:**

Moments can be defined as quantitative measures of the shape of a probability density function.


Ready? Here we go!

- The first moment works out to just be the mean of the data that you're looking at. That's it. The first moment is the mean, the average. It's that simple.
- The second moment is the variance. That's it. The second moment of the dataset is the same thing as the variance value. It might seem a little bit creepy that these things kind of fall out of the math naturally, but think about it. The variance is really based on the square of the differences from the mean, so coming up with a mathematical way of saying that variance is related to mean isn't really that much of a stretch, right. It's just that simple.
- Now when we get to the third and fourth moments, things get a little bit trickier, but they're still concepts that are easy to grasp. The third moment is called skew, and it is basically a measure of how lopsided a distribution is.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/9/5.png)

You can see in these two examples above that, if I have a longer tail on the left, now then that is a negative skew, and if I have a longer tail on the right then, that's a positive skew. The dotted lines show what the shape of a normal distribution would look like without skew. The dotted line out on the left side then I end up with a negative skew, or on the other side, a positive skew in that example. OK, so that's all skew is. It's basically stretching out the tail on one side or the other, and it is a measure of how lopsided, or how skewed a distribution is.

- The fourth moment is called kurtosis. Wow, that's a fancy word! All that really is, is how thick is the tail and how sharp is the peak. So again, it's a measure of the shape of the data distribution. Here's an example:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/9/6.png)

You can see that the higher peak values have a higher kurtosis value. The topmost curve has a higher kurtosis than the bottommost curve. It's a very subtle difference, but a difference nonetheless. It basically measures how peaked your data is.

Let's review all that: the first moment is mean, the second moment is variance, the third moment is skew, and the fourth moment is kurtosis. We already know what mean and variance are. Skew is how lopsided the data is, how stretched out one of the tails might be. Kurtosis is how peaked, how squished together the data distribution is.

### Computing moments in Python

Let's play around in Python and actually compute these moments and see how you do that. To play around with this, go ahead and open up the `Moments.ipynb`, and you can follow along with me here.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `Moments.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/Moments.ipynb

Let's again create that same normal distribution of random data. Again, we're going to make it centered around zero, with a 0.5 standard deviation and 10,000 data points, and plot that out:

```
import numpy as np 
import matplotlib.pyplot as plt 
 
vals = np.random.normal(0, 0.5, 10000) 
 
plt.hist(vals, 50) 
plt.show()
```

So again, we get a randomly generated set of data with a normal distribution around zero.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-02-01/steps/9/7.png)

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


If you want to play around with it, go ahead and, again, try to modify the distribution. Make it centered around something besides 0, and see if that actually changes anything. Should it? Well, it really shouldn't because these are all measures of the shape of the distribution, and it doesn't really say a whole lot about where that distribution is exactly. It's a measure of the shape. That's what the moments are all about. Go ahead and play around with that, try different center values, try different standard deviation values, and see what effect it has on these values, and it doesn't change at all. Of course, you'd expect things like the mean to change because you're changing the mean value, but variance, skew, maybe not. Play around, find out.

There you have percentiles and moments. Percentiles are a pretty simple concept. Moments sound hard, but it's actually pretty easy to understand how to do it, and it's easy in Python too. Now you have that under your belt. It's time to move on.