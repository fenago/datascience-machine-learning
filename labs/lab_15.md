### Introduction

In this scenario, we'll see the concept of A/B testing. We'll go through the t-test, the t-statistic, and the p-value, all useful tools for determining whether a result is actually real or a result of random variation. We'll dive into some real examples and get our hands dirty with some Python code and compute the t-statistics and p-values.

Following that, we'll look into how long you should run an experiment for before reaching a conclusion. Finally, we'll discuss the potential issues that can harm the results of your experiment and may cause you to reach the wrong conclusion.

We'll cover the following topics:

- A/B testing concepts
- T-test and p-value
- Measuring t-statistics and p-values using Python
- Determining how long to run an experiment
- A/B test gotchas

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

 ### A/B testing concepts

 If you work as a data scientist at a web company, you'll probably be asked to spend some time analyzing the results of A/B tests. These are basically controlled experiments on a website to measure the impact of a given change. So, let's talk about what A/B tests are and how they work.

#### A/B tests
If you're going to be a data scientist at a big tech web company, this is something you're going to definitely be involved in, because people need to run experiments to try different things on a website and measure the results of it, and that's actually not as straightforward as most people think it is.

What is an A/B test? Well, it's a controlled experiment that you usually run on a website, it can be applied to other contexts as well, but usually we're talking about a website, and we're going to test the performance of some change to that website, versus the way it was before.

You basically have a control set of people that see the old website, and a test group of people that see the change to the website, and the idea is to measure the difference in behavior between these two groups and use that data to actually decide whether this change was beneficial or not.

### A/B tests...


For example, I own a business that has a website, we license software to people, and right now I have a nice, friendly, orange button that people click on when they want to buy a license as shown on the left in the following figure. But what would happen if I changed the color of that button to blue, as shown on the right?

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-10/steps/3/1.png)

So in this example, if I want to find out whether blue would be better. How do I know?

I mean, intuitively, maybe that might capture people's attention more, or intuitively, maybe people are more used to seeing orange buy buttons and are more likely to click on that, I could spin that either way, right? So, my own internal biases or preconceptions don't really matter. What matters is how people react to this change on my actual website, and that's what an A/B test does.

A/B testing will split people up into people who see the orange button, and people who see the blue button, and I can then measure the behavior between these two groups and how they might differ, and make my decision on what color my buttons should be based on that data.

You can test all sorts of things with an A/B test. These include:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-10/steps/3/2.png)

### Measuring conversion for A/B testing

The first thing you need to figure out when you're designing an experiment on a website is what are you trying to optimize for? What is it that you really want to drive with this change? And this isn't always a very obvious thing. Maybe it's the amount that people spend, the amount of revenue. Well, we talked about the problems with variance in using amount spent, but if you have enough data, you can still, reach convergence on that metric a lot of times.

However, maybe that's not what you actually want to optimize for. Maybe you're actually selling some items at a loss intentionally just to capture market share. There's more complexity that goes into your pricing strategy than just top-line revenue.

Maybe what you really want to measure is profit, and that can be a very tricky thing to measure, because a lot of things cut into how much money a given product might make and those things might not always be obvious. And again, if you have loss leaders, this experiment will discount the effect that those are supposed to have. Maybe you just care about driving ad clicks on your website, or order quantities to reduce variance, maybe people are okay with that.

**Note:**

The bottom line is that you have to talk to the business owners of the area that's being tested and figure out what it is they're trying to optimize for. What are they being measured on? What is their success measured on? What are their key performance indicators or whatever the NBAs want to call it? And make sure that we're measuring the thing that it matters to them.

You can measure more than one thing at once too, you don't have to pick one, you can actually report on the effect of many different things:

- Revenue
- Profit
- Clicks
- Ad views

If these things are all moving in the right direction together, that's a very strong sign that this change had a positive impact in more ways than one. So, why limit yourself to one metric? Just make sure you know which one matters the most in what's going to be your criteria for success of this experiment ahead of time.

How to attribute conversions
Another thing to watch out for is attributing conversions to a change downstream. If the action you're trying to drive doesn't happen immediately upon the user experiencing the thing that you're testing, things get a little bit dodgy.

Let's say I change the color of a button on page A, the user then goes to page B and does something else, and ultimately buys something from page C.

Well, who gets credit for that purchase? Is it page A, or page B, or something in-between? Do I discount the credit for that conversion depending on how many clicks that person took to get to the conversion action? Do I just discard any conversion action that doesn't happen immediately after seeing that change? These are complicated things and it's very easy to produce misleading results by fudging how you account for these different distances between the conversion and the change that you're measuring.

### Variance is your enemy

Another thing that you need to really internalize is that variance is your enemy when you're running an A/B test.

A very common mistake people make who don't know what they're doing with data science is that they will put up a test on a web page, blue button versus orange button, whatever it is, run it for a week, and take the mean amount spent from each of those groups. They then say "oh look! The people with the blue button on average spent a dollar more than the people with the orange button; blue is awesome, I love blue, I'm going to put blue all over the website now!"

But, in fact, all they might have been seeing was just a random variation in purchases. They didn't have a big enough sample because people don't tend to purchase a lot. You get a lot of views but you probably don't have a lot of purchases on your website in comparison, and it's probably a lot of variance in those purchase amounts because different products cost different amounts.

So, you could very easily end up making the wrong decision that ends up costing your company money in the long run, instead of earning your company money if you don't understand the effect of variance on these results. We'll talk about some principal ways of measuring and accounting for that later in the chapter.

**Note:**

You need to make sure that your business owners understand that this is an important effect that you need to quantify and understand before making business decisions following an A/B test or any experiment that you run on the web.

Now, sometimes you need to choose a conversion metric that has less variance. It could be that the numbers on your website just mean that you would have to run an experiment for years in order to get a significant result based on something like revenue or amount spent.

Sometimes if you're looking at more than one metric, such as order amount or order quantity, that has less variance associated with it, you might see a signal on order quantity before you see a signal on revenue, for example. At the end of the day, it ends up being a judgment call. If you see a significant lift in order quantities and maybe a not-so-significant lift in revenue, then you have to say "well, I think there might be something real and beneficial going on here."

However, the only thing that statistics and data size can tell you, are probabilities that an effect is real. It's up to you to decide whether or not it's real at the end of the day. So, let's talk about how to do this in more detail.

The key takeaway here is, just looking at the differences in means isn't enough. When you're trying to evaluate the results of an experiment, you need to take the variance into account as well.

### T-test and p-value

How do you know if a change resulting from an A/B test is actually a real result of what you changed, or if it's just random variation? Well, there are a couple of statistical tools at our disposal called the t-test or t-statistic, and the p-value. Let's learn more about what those are and how they can help you determine whether an experiment is good or not.

The aim is to figure out if a result is real or not. Was this just a result of random variance that's inherent in the data itself, or are we seeing an actual, statistically significant change in behavior between our control group and our test group? T-tests and p-values are a way to compute that.

**Note:**

Remember, statistically significant doesn't really have a specific meaning. At the end of the day it has to be a judgment call. You have to pick a probability value that you're going to accept of a result being real or not. But there's always going to be a chance that it's still a result of random variation, and you have to make sure your stakeholders understand that.

### The t-statistic or t-test

Let's start with the t-statistic, also known as a t-test. It is basically a measure of the difference in behavior between these two sets, between your control and treatment group, expressed in units of standard error. It is based on standard error, which accounts for the variance inherent in the data itself, so by normalizing everything by that standard error, we get some measure of the change in behavior between these two groups that takes that variance into account.

The way to interpret a t-statistic is that a high t-value means there's probably a real difference between these two sets, whereas a low t-value means not so much difference. You have to decide what's a threshold that you're willing to accept? The sign of the t-statistic will tell you if it's a positive or negative change.

If you're comparing your control to your treatment group and you end up with a negative t-statistic, that implies that this is a bad change. You ultimate want the absolute value of that t-statistic to be large. How large a value of t-statistic is considered large? Well, that's debatable. We'll look at some examples shortly.

Now, this does assume that you have a normal distribution of behavior, and when we're talking about things like the amount people spend on a website, that's usually a decent assumption. There does tend to be a normal distribution of how much people spend.

However, there are more refined versions of t-statistics that you might want to look at for other specific situations. For example, there's something called Fisher's exact test for when you're talking about click through rates, the E-test when you're talking about transactions per user, like how many web pages do they see, and the chi-squared test, which is often relevant for when you're looking at order quantities. Sometimes you'll want to look at all of these statistics for a given experiment, and choose the one that actually fits what you're trying to do the best.

### The p-value

Now, it's a lot easier to talk about p-values than t-statistics because you don't have to think about, how many standard deviations are we talking about? What does the actual value mean? The p-value is a little bit easier for people to understand, which makes it a better tool for you to communicate the results of an experiment to the stakeholders in your business.

The p-value is basically the probability that this experiment satisfies the null hypothesis, that is, the probability that there is no real difference between the control and the treatment's behavior. A low p-value means there's a low probability of it having no effect, kind of a double negative going on there, so it's a little bit counter intuitive, but at the end of the day you just have to understand that a low p-value means that there's a high probability that your change had a real effect.

What you want to see are a high t-statistic and a low p-value, and that will imply a significant result. Now, before you start your experiment, you need to decide what your threshold for success is going to be, and that means deciding the threshold with the people in charge of the business.

So, what p-value are you willing to accept as a measure of success? Is it 1 percent? Is it 5 percent? And again, this is basically the likelihood that there is no real effect, that it's just a result of random variance. It is just a judgment call at the end of the day. A lot of times people use 1 percent, sometimes they use 5 percent if they're feeling a little bit riskier, but there's always going to be that chance that your result was just spurious, random data that came in.

However, you can choose the probability that you're willing to accept as being likely enough that this is a real effect, that's worth rolling out into production.

When your experiment is over, and we'll talk about when you declare an experiment to be over later, you want to measure your p-value. If it's less than the threshold you decided upon, then you can reject the null hypothesis and you can say "well, there's a high likelihood that this change produced a real positive or negative result."

If it is a positive result then you can roll that change out to the entire site and it is no longer an experiment, it is part of your website that will hopefully make you more and more money as time goes on, and if it's a negative result, you want to get rid of it before it costs you any more money.

**Note:**

Remember, there is a real cost to running an A/B test when your experiment has a negative result. So, you don't want to run it for too long because there's a chance you could be losing money.

This is why you want to monitor the results of an experiment on a daily basis, so if there are early indications that the change is making a horrible impact to the website, maybe there's a bug in it or something that's horrible, you can pull the plug on it prematurely if necessary, and limit the damage.

Let's go to an actual example and see how you might measure t-statistics and p-values using Python.

### Measuring t-statistics and p-values using Python

Let's fabricate some experimental data and use the t-statistic and p-value to determine whether a given experimental result is a real effect or not. We're going to actually fabricate some fake experimental data and run t-statistics and p-values on them, and see how it works and how to compute it in Python.

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `TTest.ipynb` in the `work` folder.

You can also open the Jupyter Notebook at https://[[HOST_SUBDOMAIN]]-8888-[[KATACODA_HOST]].environments.katacoda.com/notebooks/work/TTest.ipynb


#### Running A/B test on some experimental data
Let's imagine that we're running an A/B test on a website and we have randomly assigned our users into two groups, group A and group B. The A group is going to be our test subjects, our treatment group, and group B will be our control, basically the way the website used to be. We'll set this up with the following code:

```
import numpy as np 
from scipy import stats 
 
A = np.random.normal(25.0, 5.0, 10000) 
B = np.random.normal(26.0, 5.0, 10000) 
 
stats.ttest_ind(A, B) 
```

In this code example, our treatment group (A) is going to have a randomly distributed purchase behavior where they spend, on average, $25 per transaction, with a standard deviation of five and ten thousand samples, whereas the old website used to have a mean of $26 per transaction with the same standard deviation and sample size. We're basically looking at an experiment that had a negative result. All you have to do to figure out the t-statistic and the p-value is use this handy stats.ttest_ind method from scipy. What you do is, you pass it in your treatment group and your control group, and out comes your t-statistic as shown in the output here:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-10/steps/9/1.jpg)

In this case, we have a t-statistic of -14. The negative indicates that it is a negative change, this was a bad thing. And the p-value is very, very small. So, that implies that there is an extremely low probability that this change is just a result of random chance.

**Note:**

Remember that in order to declare significance, we need to see a high t-value t-statistic, and a low p-value.

That's exactly what we're seeing here, we're seeing -14, which is a very high absolute value of the t-statistic, negative indicating that it's a bad thing, and an extremely low P-value, telling us that there's virtually no chance that this is just a result of random variation.

If you saw these results in the real world, you would pull the plug on this experiment as soon as you could.



**When there's no real difference between the two groups**

Just as a sanity check, let's go ahead and change things so that there's no real difference between these two groups. So, I'm going to change group B, the control group in this case, to be the same as the treatment, where the mean is 25, the standard deviation is unchanged, and the sample size is unchanged as shown here:

```
B = np.random.normal(25.0, 5.0, 10000) 
 
stats.ttest_ind(A, B) 
```

If we go ahead and run this, you can see our t-test ends up being below one now:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-10/steps/9/2.jpg)

Remember this is in terms of standard deviation. So this implies that there's probably not a real change there unless we have a much higher p-value as well, over 30 percent.

Now, these are still relatively high numbers. You can see that random variation can be kind of an insidious thing. This is why you need to decide ahead of time what would be an acceptable limit for p-value.

You know, you could look at this after the fact and say, "30 percent odds, you know, that's not so bad, we can live with that," but, no. I mean, in reality and practice you want to see p-values that are below 5 percent, ideally below 1 percent, and a value of 30 percent means it's actually not that strong of a result. So, don't justify it after the fact, go into your experiment in knowing what your threshold is.

### Does the sample size make a difference?

Let's do some changes in the sample size. We're creating these sets under the same conditions. Let's see if we actually get a difference in behavior by increasing the sample size.

Sample size increased to six-digits
So, we're going to go from 10000 to 100000 samples as shown here:

```
A = np.random.normal(25.0, 5.0, 100000) 
B = np.random.normal(25.0, 5.0, 100000) 
 
stats.ttest_ind(A, B)
```

You can see in the following output that actually the p-value got a little bit lower and the t-test a little bit larger, but it's still not enough to declare a real difference. It's actually going in the direction you wouldn't expect it to go? Kind of interesting!

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-10/steps/9/3.jpg)

But these are still high values. Again, it's just the effect of random variance, and it can have more of an effect than you realize. Especially on a website when you're talking about order amounts.

Sample size increased seven-digits
Let's actually increase the sample size to 1000000, as shown here:

```
A = np.random.normal(25.0, 5.0, 1000000) 
B = np.random.normal(25.0, 5.0, 1000000) 
 
stats.ttest_ind(A, B) 
```

Here is the result:

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-10/steps/9/4.jpg)

What does that do? Well, now, we're back under 1 for the t-statistic, and our value's around 35 percent.

We're seeing these kind of fluctuations a little bit in either direction as we increase the sample size. This means that going from 10,000 samples to 100,000 to 1,000,000 isn't going to change your result at the end of the day. And running experiments like this is a good way to get a good gut feel as to how long you might need to run an experiment for. How many samples does it actually take to get a significant result? And if you know something about the distribution of your data ahead of time, you can actually run these sorts of models.

### A/A testing

If we were to compare the set to itself, this is called an A/A test as shown in the following code example:

```
stats.ttest_ind(A, A) 
```

We can see in the following output, a t-statistic of 0 and a p-value of 1.0 because there is in fact no difference whatsoever between these sets.

![](https://github.com/fenago/katacoda-scenarios/raw/master/datascience-machine-learning/datascience-machine-learning-chapter-10/steps/9/5.jpg)

Now, if you were to run that using real website data where you were looking at the same exact people and you saw a different value, that indicates there's a problem in the system itself that runs your testing. At the end of the day, like I said, it's all a judgment call.

Go ahead and play with this, see what the effect of different standard deviations has on the initial datasets, or differences in means, and different sample sizes. I just want you to dive in, play around with these different datasets and actually run them, and see what the effect is on the t-statistic and the p-value. And hopefully that will give you a more gut feel of how to interpret these results.

**Note:**

Again, the important thing to understand is that you're looking for a large t-statistic and a small p-value. P-value is probably going to be what you want to communicate to the business. And remember, lower is better for p-value, you want to see that in the single digits, ideally below 1 percent before you declare victory.

We'll talk about A/B tests some more in the remainder of the chapter. SciPy makes it really easy to compute t-statistics and p-values for a given set of data, so you can very easily compare the behavior between your control and treatment groups, and measure what the probability is of that effect being real or just a result of random variation. Make sure you are focusing on those metrics and you are measuring the conversion metric that you care about when you're doing those comparisons.

### Determining how long to run an experiment for

How long do you run an experiment for? How long does it take to actually get a result? At what point do you give up? Let's talk about that in more detail.

If someone in your company has developed a new experiment, a new change that they want to test, then they have a vested interest in seeing that succeed. They put a lot of work and time into it, and they want it to be successful. Maybe you've gone weeks with the testing and you still haven't reached a significant outcome on this experiment, positive or negative. You know that they're going to want to keep running it pretty much indefinitely in the hope that it will eventually show a positive result. It's up to you to draw the line on how long you're willing to run this experiment for.

How do I know when I'm done running an A/B test? I mean, it's not always straightforward to predict how long it will take before you can achieve a significant result, but obviously if you have achieved a significant result, if your p-value has gone below 1 percent or 5 percent or whatever threshold you've chosen, and you're done.

At that point you can pull the plug on the experiment and either roll out the change more widely or remove it because it was actually having a negative effect. You can always tell people to go back and try again, use what they learned from the experiment to maybe try it again with some changes and soften the blow a little bit.

The other thing that might happen is it's just not converging at all. If you're not seeing any trends over time in the p-value, it's probably a good sign that you're not going to see this converge anytime soon. It's just not going to have enough of an impact on behavior to even be measurable, no matter how long you run it.

In those situations, what you want to do every day is plot on a graph for a given experiment the p-value, the t-statistic, whatever you're using to measure the success of this experiment, and if you're seeing something that looks promising, you will see that p-value start to come down over time. So, the more data it gets, the more significant your results should be getting.

Now, if you instead see a flat line or a line that's all over the place, that kind of tells you that that p-value's not going anywhere, and it doesn't matter how long you run this experiment, it's just not going to happen. You need to agree up front that in the case where you're not seeing any trends in p-values, what's the longest you're willing to run this experiment for? Is it two weeks? Is it a month?

**Note:**

Another thing to keep in mind is that having more than one experiment running on the site at once can conflate your results.

Time spent on experiments is a valuable commodity, you can't make more time in the world. You can only really run as many experiments as you have time to run them in a given year. So, if you spend too much time running one experiment that really has no chance of converging on a result, that's an opportunity you've missed to run another potentially more valuable experiment during that time that you are wasting on this other one.

It's important to draw the line on experiment links, because time is a very precious commodity when you're running A/B tests on a website, at least as long as you have more ideas than you have time, which hopefully is the case. Make sure you go in with agreed upper bounds on how long you're going to spend testing a given experiment, and if you're not seeing trends in the p-value that look encouraging, it's time to pull the plug at that point.

### A/B test gotchas


An important point I want to make is that the results of an A/B test, even when you measure them in a principled manner using p-values, is not gospel. There are many effects that can actually skew the results of your experiment and cause you to make the wrong decision. Let's go through a few of these and let you know how to watch out for them. Let's talk about some gotchas with A/B tests.

It sounds really official to say there's a p-value of 1 percent, meaning there's only a 1 percent chance that a given experiment was due to spurious results or random variation, but it's still not the be-all and end-all of measuring success for an experiment. There are many things that can skew or conflate your results that you need to be aware of. So, even if you see a p-value that looks very encouraging, your experiment could still be lying to you, and you need to understand the things that can make that happen so you don't make the wrong decisions.

**Note:**

Remember, correlation does not imply causation.

Even with a well-designed experiment, all you can say is there is some probability that this effect was caused by this change you made.

At the end of the day, there's always going to be a chance that there was no real effect, or you might even be measuring the wrong effect. It could still be random chance, there could be something else going on, it's your duty to make sure the business owners understand that these experimental results need to be interpreted, they need to be one piece of their decision.

They can't be the be-all and end-all that they base their decision on because there is room for error in the results and there are things that can skew those results. And if there's some larger business objective to this change, beyond just driving short-term revenue, that needs to be taken into account as well.

Novelty effects
One problem is novelty effects. One major Achilles heel of an A/B test is the short time frame over which they tend to be run, and this causes a couple of problems. First of all, there might be longer-term effects to the change, and you're not going to measure those, but also, there is a certain effect to just something being different on the website.

For instance, maybe your customers are used to seeing the orange buttons on the website all the time, and if a blue button comes up and it catches their attention just because it's different. However, as new customers come in who have never seen your website before, they don't notice that as being different, and over time even your old customers get used to the new blue button. It could very well be that if you were to make this same test a year later, there would be no difference. Or maybe they'd be the other way around.

I could very easily see a situation where you test orange button versus blue button, and in the first two weeks the blue button wins. People buy more because they are more attracted to it, because it's different. But a year goes by, I could probably run another web lab that puts that blue button against an orange button and the orange button would win, again, simply because the orange button is different, and it's new and catches people's attention just for that reason alone.

For that reason, if you do have a change that is somewhat controversial, it's a good idea to rerun that experiment later on and see if you can actually replicate its results. That's really the only way I know of to account for novelty effects; actually measure it again when it's no longer novel, when it's no longer just a change that might capture people's attention simply because it's different.

And this, I really can't understate the importance of understanding this. This can really skew a lot of results, it biases you to attributing positive changes to things that don't really deserve it. Being different in and of itself is not a virtue; at least not in this context.

### Seasonal effects

If you're running an experiment over Christmas, people don't tend to behave the same during Christmas as they do the rest of the year. They definitely spend their money differently during that season, they're spending more time with their families at home, and they might be a little bit, kind of checked out of work, so people have a different frame of mind.

It might even be involved with the weather, during the summer people behave differently because it's hot out they're feeling kind of lazy, they're on vacation more often. Maybe if you happen to do your experiment during the time of a terrible storm in a highly populated area that could skew your results as well.

Again, just be cognizant of potential seasonal effects, holidays are a big one to be aware of, and always take your experience with a grain of salt if they're run during a period of time that's known to have seasonality.

You can determine this quantitatively by actually looking at the metric you're trying to measure as a success metric, be it, whatever you're calling your conversion metric, and look at its behavior over the same time period last year. Are there seasonal fluctuations that you see every year? And if so, you want to try to avoid running your experiment during one of those peaks or valleys.


### Selection bias

Another potential issue that can skew your results is selection bias. It's very important that customers are randomly assigned to either your control or your treatment groups, your A or B group.

However, there are subtle ways in which that random assignment might not be random after all. For example, let's say that you're hashing your customer IDs to place them into one bucket or the other. Maybe there's some subtle bias between how that hash function affects people with lower customer IDs versus higher customer IDs. This might have the effect of putting all of your longtime, more loyal customers into the control group, and your newer customers who don't know you that well into your treatment group.

What you end up measuring then is just a difference in behavior between old customers and new customers as a result. It's very important to audit your systems to make sure there is no selection bias in the actual assignment of people to the control or treatment group.

You also need to make sure that assignment is sticky. If you're measuring the effect of a change over an entire session, you want to measure if they saw a change on page A but, over on page C they actually did a conversion, you have to make sure they're not switching groups in between those clicks. So, you need to make sure that within a given session, people remain in the same group, and how to define a session can become kind of nebulous as well.

Now, these are all issues that using an established off-the-shelf framework like Google Experiments or Optimizely or one of those guys can help with so that you're not reinventing the wheel on all these problems. If your company does have a homegrown, in-house solution because they're not comfortable with sharing that data with outside companies, then it's worth auditing whether there is selection bias or not.

### Auditing selection bias issues

One way for auditing selection bias issues is running what's called an A/A test, like we saw earlier. So, if you actually run an experiment where there is no difference between the treatment and control, you shouldn't see a difference in the end result. There should not be any sort of change in behavior when you're comparing those two things.

An A/A test can be a good way of testing your A/B framework itself and making sure there's no inherent bias or other problems, for example, session leakage and whatnot, that you need to address.

Data pollution
Another big problem is data pollution. We talked at length about the importance of cleaning your input data, and it's especially important in the context of an A/B test. What would happen if you have a robot, a malicious crawler that's crawling through your website all the time, doing an unnatural amount of transactions? What if that robot ends up getting either assigned to the treatment or the control?

That one robot could skew the results of your experiment. It's very important to study the input going into your experiment and look for outliers, then analyze what those outliers are, and whether they should they be excluded. Are you actually letting some robots leak into your measurements and are they skewing the results of your experiment? This is a very, very common problem, and something you need to be cognizant of.

There are malicious robots out there, there are people trying to hack into your website, there are benign scrapers just trying to crawl your website for search engines or whatnot. There are all sorts of weird behaviors going on with a website, and you need to filter out those and get at the people who are really your customers and not these automated scripts. That can actually be a very challenging problem. Yet another reason to use off-the-shelf frameworks like Google Analytics, if you can.

### Attribution errors

We talked briefly about attribution errors earlier. This is if you are actually using downstream behavior from a change, and that gets into a gray area.

You need to understand how you're actually counting those conversions as a function of distance from the thing that you changed and agree with your business stakeholders upfront as to how you're going to measure those effects. You also need to be aware of if you're running multiple experiments at once; will they conflict with one another? Is there a page flow where someone might actually encounter two different experiments within the same session?

If so, that's going to be a problem and you have to apply your judgment as to whether these changes actually could interfere with each other in some meaningful way and affect the customers' behavior in some meaningful way. Again, you need to take these results with a grain of salt. There are a lot of things that can skew results and you need to be aware of them. Just be aware of them and make sure your business owners are also aware of the limitations of A/B tests and all will be okay.

Also, if you're not in a position where you can actually devote a very long amount of time to an experiment, you need to take those results with a grain of salt and ideally retest them later on during a different time period.

