<img align="right" src="../images/logo-small.png">


Lab : Apache Spark - Machine Learning on Big Data - Part 2
-------------------------------------


Everyone talks about big data, and odds are you might be working for a company that does in fact have big data to process. Big data meaning that you can't actually control it all, you can't actually wrangle it all on just one system. You need to actually compute it using the resources of an entire cloud, a cluster of computing resources. And that's where Apache Spark comes in. Apache Spark is a very powerful tool for managing big data, and doing machine learning on large Datasets. By the end of the chapter, you will have an in-depth knowledge of the following topics:

- Working with Spark
- Decision Trees in Spark
- K-Means Clustering in Spark

### Terminal

Now, move in the directory which contains the source code.

`cd ~/work/datascience-machine-learning`

**Note:**
- The supplied commands in the next steps MUST be run from your `datascience-machine-learning` directory. 
- There should be terminal opened already. You can also open New terminal by Clicking `File` > `New` > `Terminal` from the top menu.

### TF-IDF

So, our final example of MLlib is going to be using something called Term Frequency Inverse Document Frequency, or TF-IDF, which is the fundamental building block of many search algorithms. As usual, it sounds complicated, but it's not as bad as it sounds.

So, first, let's talk about the concepts of TF-IDF, and how we might go about using that to solve a search problem. And what we're actually going to do with TF-IDF is create a rudimentary search engine for Wikipedia using Apache Spark in MLlib. How awesome is that? Let's get started.

TF-IDF stands for Term Frequency and Inverse Document Frequency, and these are basically two metrics that are closely interrelated for doing search and figuring out the relevancy of a given word to a document, given a larger body of documents. So, for example, every article on Wikipedia might have a term frequency associated with it, every page on the Internet could have a term frequency associated with it for every word that appears in that document. Sounds fancy, but, as you'll see, it's a fairly simple concept.

- **All Term Frequency** means is how often a given word occurs in a given document. So, within one web page, within one Wikipedia article, within one whatever, how common is a given word within that document? You know, what is the ratio of that word's occurrence rate throughout all the words in that document? That's it. That's all term frequency is.
- **Document frequency** is the same idea, but this time it is the frequency of that word across the entire corpus of documents. So, how often does this word occur throughout all of the documents that I have, all the web pages, all of the articles on Wikipedia, whatever. For example, common words like "a" or "the" would have a very high document frequency, and I would expect them to also have a very high term frequency, but that doesn't necessarily mean they're relevant to a given document.

You can kind of see where we're going with this. So, let's say we have a very high term frequency and a very low document frequency for a given word. The ratio of these two things can give me a measure of the relevance of that word to the document. So, if I see a word that occurs very often in a given document, but not very often in the overall space of documents, then I know that this word probably conveys some special meaning to this particular document. It might convey what this document is actually about.

So, that's TF-IDF. It just stands for Term Frequency x Inverse Document Frequency, which is just a fancy way of saying term frequency over document frequency, which is just a fancy way of saying how often does this word occur in this document compared to how often it occurs in the entire body of documents? It's that simple.

### TF-IDF in practice

In practice, there are a few little nuances to how we use this. For example, we use the actual log of the inverse document frequency instead of the raw value, and that's because word frequencies in reality tend to be distributed exponentially. So, by taking the log, we end up with a slightly better weighting of words, given their overall popularity. There are some limitations to this approach, obviously, one is that we basically assume a document is nothing more than a bagful of words, we assume there are no relationships between the words themselves. And, obviously, that's not always the case, and actually parsing them out can be a good part of the work, because you have to deal with things like synonyms and various tenses of words, abbreviations, capitalizations, misspellings, and so on. This gets back to the idea of cleaning your data being a large part of your job as a data scientist, and it's especially true when you're dealing with natural language processing stuff. Fortunately, there are some libraries out there that can help you with this, but it is a real problem and it will affect the quality of your results.

Another implementation trick that we use with TF-IDF is, instead of storing actual string words with their term frequencies and inverse document frequency, to save space and make things more efficient, we actually map every word to a numerical value, a hash value we call it. The idea is that we have a function that can take any word, look at its letters, and assign that, in some fairly well-distributed manner, to a set of numbers in a range. That way, instead of using the word "represented", we might assign that a hash value of 10, and we can then refer to the word "represented" as "10" from now on. Now, if the space of your hash values isn't large enough, you could end up with different words being represented by the same number, which sounds worse than it is. But, you know, you want to make sure that you have a fairly large hash space so that is unlikely to happen. Those are called hash collisions. They can cause issues, but, in reality, there's only so many words that people commonly use in the English language. You can get away with 100,000 or so and be just fine.

Doing this at scale is the hard part. If you want to do this over all of Wikipedia, then you're going to have to run this on a cluster. But for the sake of argument, we are just going to run this on our own desktop for now, using a small sample of Wikipedia data.

### Using TF- IDF

How do we turn that into an actual search problem? Once we have TF-IDF, we have this measure of each word's relevancy to each document. What do we do with it? Well, one thing you could do is compute TF-IDF for every word that we encounter in the entire body of documents that we have, and then, let's say we want to search for a given term, a given word. Let's say we want to search for "what Wikipedia article in my set of Wikipedia articles is most relevant to Gettysburg?" I could sort all the documents by their TF-IDF score for Gettysburg, and just take the top results, and those are my search results for Gettysburg. That's it. Just take your search word, compute TF-IDF, take the top results. That's it.

Obviously, in the real world there's a lot more to search than that. Google has armies of people working on this problem and it's way more complicated in practice, but this will actually give you a working search engine algorithm that produces reasonable results. Let's go ahead and dive in and see how it all works.

### Searching wikipedia with Spark MLlib

We're going to build an actual working search algorithm for a piece of Wikipedia using Apache Spark in MLlib, and we're going to do it all in less than 50 lines of code. This might be the coolest thing we do in this entire book!

Go into your course materials and open up the TF-IDF.py script, and that should open up Canopy with the following code:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-09-02/steps/22/1.png)

Now, step back for a moment and let it sink in that we're actually creating a working search algorithm, along with a few examples of using it in less than 50 lines of code here, and it's scalable. I could run this on a cluster. It's kind of amazing. Let's step through the code.

## Import statements

We're going to start by importing the SparkConf and SparkContext libraries that we need for any Spark script that we run in Python, and then we're going to import HashingTF and IDF using the following commands.

```
from pyspark import SparkConf, SparkContext 
from pyspark.mllib.feature import HashingTF 
from pyspark.mllib.feature import IDF 
```

So, this is what computes the term frequencies (TF) and inverse document frequencies (IDF) within our documents.

#### Creating the initial RDD
We'll start off with our boilerplate Spark stuff that creates a local SparkConfiguration and a SparkContext, from which we can then create our initial RDD.

```
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF") 
sc = SparkContext(conf = conf) 
```

Next, we're going to use our SparkContext to create an RDD from subset-small.tsv.

```
rawData = sc.textFile("e:/sundog-consult/Udemy/DataScience/subset-small.tsv") 
```

This is a file containing tab-separated values, and it represents a small sample of Wikipedia articles. Again, you'll need to change your path as shown in the preceding code as necessary for wherever you installed the course materials for this book.

That gives me back an RDD where every document is in each line of the RDD. The tsv file contains one entire Wikipedia document on every line, and I know that each one of those documents is split up into tabular fields that have various bits of metadata about each article.


The next thing I'm going to do is split those up:

```
fields = rawData.map(lambda x: x.split("\t")) 
```

I'm going to split up each document based on their tab delimiters into a Python list, and create a new fields RDD that, instead of raw input data, now contains Python lists of each field in that input data.

Finally, I'm going to map that data, take in each list of fields, extract field number three x[3], which I happen to know is the body of the article itself, the actual article text, and I'm in turn going to split that based on spaces:

```
documents = fields.map(lambda x: x[3].split(" ")) 
```

What x[3] does is extract the body of the text from each Wikipedia article, and split it up into a list of words. My new documents RDD has one entry for every document, and every entry in that RDD contains a list of words that appear in that document. Now, we actually know what to call these documents later on when we're evaluating the results.

I'm also going to create a new RDD that stores the document names:

```
documentNames = fields.map(lambda x: x[1]) 
```

All that does is take that same fields RDD and uses this map function to extract the document name, which I happen to know is in field number one.

So, I now have two RDDs, documents, which contains lists of words that appear in each document, and documentNames, which contains the name of each document. I also know that these are in the same order, so I can actually combine these together later on to look up the name for a given document.

### Creating and transforming a HashingTF object

Now, the magic happens. The first thing we're going to do is create a HashingTF object, and we're going to pass in a parameter of 100,000. This means that I'm going to hash every word into one of 100,000 numerical values:

```
hashingTF = HashingTF(100000)  
```

Instead of representing words internally as strings, which is very inefficient, it's going to try to, as evenly as possible, distribute each word to a unique hash value. I'm giving it up to 100,000 hash values to choose from. Basically, this is mapping words to numbers at the end of the day.

Next, I'm going to call transform on hashingTF with my actual RDD of documents:

```
tf = hashingTF.transform(documents) 
```

That's going to take my list of words in every document and convert it to a list of hash values, a list of numbers that represent each word instead.

This is actually represented as a sparse vector at this point to save even more space. So, not only have we converted all of our words to numbers, but we've also stripped out any missing data. In the event that a word does not appear in a document where you're not storing the fact that word does not appear explicitly, it saves even more space.

#### Computing the TF-IDF score
To actually compute the TF-IDF score for each word in each document, we first cache this tf RDD.

```
tf.cache() 
```

We do that because we're going to use it more than once. Next, we use IDF(minDocFreq=2), meaning that we're going to ignore any word that doesn't appear at least twice:

```
idf = IDF(minDocFreq=2).fit(tf) 
```

We call fit on tf, and then in the next line we call transform on tf:

```
tfidf = idf.transform(tf) 
```

What we end up with here is an RDD of the TF-IDF score for each word in each document.

### Using the Wikipedia search engine algorithm

Let's try and put the algorithm to use. Let's try to look up the best article for the word Gettysburg. If you're not familiar with US history, that's where Abraham Lincoln gave a famous speech. So, we can transform the word Gettysburg into its hash value using the following code:

```
gettysburgTF = hashingTF.transform(["Gettysburg"]) 
gettysburgHashValue = int(gettysburgTF.indices[0]) 
```

We will then extract the TF-IDF score for that hash value into a new RDD for each document:

```
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])  
```

What this does is extract the TF-IDF score for Gettysburg, from the hash value it maps to for every document, and stores that in this gettysburgRelevance RDD.

We then combine that with the documentNames so we can see the results:

```
zippedResults = gettysburgRelevance.zip(documentNames)  
```

Finally, we can print out the answer:

```
print ("Best document for Gettysburg is:") 
print (zippedResults.max()) 
```

### Running the algorithm

So, let's go run that and see what happens. As usual, to run the Spark script, we're not going to just hit the play icon. We have to go to Tools>Canopy Command Prompt. In the Command Prompt that opens up, we will type in spark-submit TF-IDF.py, and off it goes.

#### Run Code
Now, run the python code by running: `python TF-IDF.py`

We are asking it to chunk through quite a bit of data, even though it's a small sample of Wikipedia it's still a fair chunk of information, so it might take a while. Let's see what comes back for the best document match for Gettysburg, what document has the highest TF-IDF score?

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-09-02/steps/22/2.png)

It's Abraham Lincoln! Isn't that awesome? We just made an actual search engine that actually works, in just a few lines of code.

And there you have it, an actual working search algorithm for a little piece of Wikipedia using Spark in MLlib and TF-IDF. And the beauty is we can actually scale that up to all of Wikipedia if we wanted to, if we had a cluster large enough to run it.

Hopefully we got your interest up there in Spark, and you can see how it can be applied to solve what can be pretty complicated machine learning problems in a distributed manner. So, it's a very important tool, and I want to make sure you don't get through this book on data science without at least knowing the concepts of how Spark can be applied to big data problems. So, when you need to move beyond what one computer can do, remember, Spark is at your disposal.

This chapter was originally produced for Spark 1, so let's talk about what's new in Spark 2, and what new capabilities exist in MLlib now.

So, the main thing with Spark 2 is that they moved more and more toward Dataframes and Datasets. Datasets and Dataframes are kind of used interchangeably sometimes. Technically a dataframe is a Dataset of row objects, they're kind of like RDDs, but the only difference is that, whereas an RDD just contains unstructured data, a Dataset has a defined schema to it.

A Dataset knows ahead of time exactly what columns of information exists in each row, and what types those are. Because it knows about the actual structure of that Dataset ahead of time, it can optimize things more efficiently. It also lets us think of the contents of this Dataset as a little, mini database, well, actually, a very big database if it's on a cluster. That means we can do things like issue SQL queries on it.

This creates a higher-level API with which we can query and analyze massive Datasets on a Spark cluster. It's pretty cool stuff. It's faster, it has more opportunities for optimization, and it has a higher-level API that's often easier to work with.

### How Spark 2.0 MLlib works

Going forward in Spark 2.0, MLlib is pushing dataframes as its primary API. This is the way of the future, so let's take a look at how it works. I've gone ahead and opened up the SparkLinearRegression.py file in Canopy, as shown in the following figure, so let's walk through it a little bit:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-09-02/steps/26/1.png)

As you see, for one thing, we're using ml instead of MLlib, and that's because the new dataframe-based API is in there.

#### Implementing linear regression
In this example, what we're going to do is implement linear regression, and linear regression is just a way of fitting a line to a set of data. What we're going to do in this exercise is take a bunch of fabricated data that we have in two dimensions, and try to fit a line to it with a linear model.

We're going to separate our data into two sets, one for building the model and one for evaluating the model, and we'll compare how well this linear model does at actually predicting real values. First of all, in Spark 2, if you're going to be doing stuff with the SparkSQL interface and using Datasets, you've got to be using a SparkSession object instead of a SparkContext. To set one up, you do the following:

```
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("LinearRegression").getOrCreate() 
```

**Note:**

**Note:** that the middle bit is only necessary on Windows and in Spark 2.0. It kind of works around a little bug that they have, to be honest. So, if you're on Windows, make sure you have a C:/temp folder. If you want to run this, go create that now if you need to. If you're not on Windows, you can delete that whole middle section to leave: spark = SparkSession.builder.appName("LinearRegression").getOrCreate().

Okay, so you can say spark, give it an appName and getOrCreate().

This is interesting, because once you've created a Spark session, if it terminates unexpectedly, you can actually recover from that the next time that you run it. So, if we have a checkpoint directory, it can actually restart where it left off using getOrCreate.

### Using the Spark 2.0 DataFrame API for MLlib


Now, we're going to use this regression.txt file that I have included with the course materials:

```
inputLines = spark.sparkContext.textFile("regression.txt")  
```

That is just a text file that has comma-delimited values of two columns, and they're just two columns of, more or less randomly, linearly correlated data. It can represent whatever you want. Let's imagine that it represents heights and weights, for example. So, the first column might represent heights, the second column might represent weights.

**Note:**

In the lingo of machine learning, we talk about labels and features, where labels are usually the thing that you're trying to predict, and features are a set of known attributes of the data that you use to make a prediction from.


In this example, maybe heights are the labels and the features are the weights. Maybe we're trying to predict heights based on your weight. It can be anything, it doesn't matter. This is all normalized down to data between -1 and 1. There's no real meaning to the scale of the data anywhere, you can pretend it means anything you want, really.

To use this with MLlib, we need to transform our data into the format it expects:

```
data = inputLines.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1])))) 
```

The first thing we're going to do is split that data up with this map function that just splits each line into two distinct values in a list, and then we're going to map that to the format that MLlib expects. That's going to be a floating point label, and then a dense vector of the feature data.

In this case, we only have one bit of feature data, the weight, so we have a vector that just has one thing in it, but even if it's just one thing, the MLlib linear regression model requires a dense vector there. This is like a labeledPoint in the older API, but we have to do it the hard way here.

Next, we need to actually assign names to those columns. Here's the syntax for doing that:

```
colNames = ["label", "features"] 
df = data.toDF(colNames) 
```

We're going to tell MLlib that these two columns in the resulting RDD actually correspond to the label and the features, and then I can convert that RDD to a DataFrame object. At this point, I have an actual dataframe or, if you will, a Dataset that contains two columns, label and features, where the label is a floating point height, and the features column is a dense vector of floating point weights. That is the format required by MLlib, and MLlib can be pretty picky about this stuff, so it's important that you pay attention to these formats.


Now, like I said, we're going to split our data in half.

```
trainTest = df.randomSplit([0.5, 0.5]) 
trainingDF = trainTest[0] 
testDF = trainTest[1] 
```

We're going to do a 50/50 split between training data and test data. This returns back two dataframes, one that I'm going to use to actually create my model, and one that I'm going to use to evaluate my model.

I will next create my actual linear regression model with a few standard parameters here that I've set.

```
lir = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
``` 

We're going to call lir = LinearRegression, and then I will fit that model to the set of data that I held aside for training, the training data frame:

```
model = lir.fit(trainingDF) 
```

That gives me back a model that I can use to make predictions from.

Let's go ahead and do that.

```
fullPredictions = model.transform(testDF).cache() 
```

I will call model.transform(testDF), and what that's going to do is predict the heights based on the weights in my testing Dataset. I actually have the known labels, the actual, correct heights, and this is going to add a new column to that dataframe called predictions, that has the predicted values based on that linear model.


I'm going to cache those results, and now I can just extract them and compare them together. So, let's pull out the prediction column, just using select like you would in SQL, and then I'm going to actually transform that dataframe and pull out the RDD from it, and use that to map it to just a plain old RDD full of floating point heights in this case:

```
predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0]) 
```

These are the predicted heights. Next, we're going to get the actual heights from the label column:

```
labels = fullPredictions.select("label").rdd.map(lambda x: x[0]) 
```

Finally, we can zip them back together and just print them out side by side and see how well it does:

```
predictionAndLabel = predictions.zip(labels).collect() 
 
for prediction in predictionAndLabel: 
    print(prediction) 
 
spark.stop() 
```

This is kind of a convoluted way of doing it; I did this to be more consistent with the previous example, but a simpler approach would be to just actually select prediction and label together into a single RDD that maps out those two columns together and then I don't have to zip them up, but either way it works. You'll also note that right at the end there we need to stop the Spark session.

So let's see if it works. Let's go up to Tools, Canopy Command Prompt, and we'll type in spark-submit SparkLinearRegression.py and let's see what happens.


#### Run Code
Now, run the python code by running: `python SparkLinearRegression.py`

There's a little bit more upfront time to actually run these APIs with Datasets, but once they get going, they're very fast. Alright, there you have it.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-09-02/steps/26/2.png)

Here we have our actual and predicted values side by side, and you can see that they're not too bad. They tend to be more or less in the same ballpark. There you have it, a linear regression model in action using Spark 2.0, using the new dataframe-based API for MLlib. More and more, you'll be using these APIs going forward with MLlib in Spark, so make sure you opt for these when you can. Alright, that's MLlib in Spark, a way of actually distributing massive computing tasks across an entire cluster for doing machine learning on big Datasets. So, good skill to have. Let's move on.
