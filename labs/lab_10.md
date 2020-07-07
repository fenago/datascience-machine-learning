<img align="right" src="../images/logo-small.png">


Lab : Recommender Systems
-------------------------------------


Let's talk about my personal area of expertise—recommender systems, so systems that can recommend stuff to people based on what everybody else did. We'll look at some examples of this and a couple of ways to do it. Specifically, two techniques called user-based and item-based collaborative filtering. So, let's dive in.

I spent most of my career at amazon.com and imdb.com, and a lot of what I did there was developing recommender systems; things like people who bought this also bought, or recommended for you, and things that did movie recommendations for people. So, this is something I know a lot about personally, and I hope to share some of that knowledge with you. We'll walk through, step by step, covering the following topics:

- What are recommender systems?
- User-based collaborative filtering
- Item-based collaborative filtering
- Finding movie similarities
- Making movie recommendations to people
- Improving the recommender's results

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/datascience-machine-learning` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab_


 ### What are recommender systems?

 Well, like I said Amazon is a great example, and one I'm very familiar with. So, if you go to their recommendations section, as shown in the following image, you can see that it will recommend things that you might be interested in purchasing based on your past behavior on the site.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/3/1.png)

The recommender system might include things that you've rated, or things that you bought, and other data as well. I can't go into the details because they'll hunt me down, and you know, do bad things to me. But, it's pretty cool. You can also think of the people who bought this also bought feature on Amazon as a form of recommender system.

The difference is that the recommendations you're seeing on your Amazon recommendations page are based on all of your past behavior, whereas people who bought this also bought or people who viewed this also viewed, things like that, are just based on the thing you're looking at right now, and showing you things that are similar to it that you might also be interested in. And, it turns out, what you're doing right now is probably the strongest signal of your interest anyhow.


Another example is from Netflix, as shown in the following image (the following image is a screenshot from Netflix):

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/3/2.png)

They have various features that try to recommend new movies or other movies you haven't seen yet, based on the movies that you liked or watched in the past as well, and they break that down by genre. They have kind of a different spin on things, where they try to identify the genres or the types of movies that they think you're enjoying the most and they then show you more results from those genres. So, that's another example of a recommender system in action.

The whole point of it is to help you discover things you might not know about before, so it's pretty cool. You know, it gives individual movies, or books, or music, or whatever, a chance to be discovered by people who might not have heard about them before. So, you know, not only is it cool technology, it also kind of levels the playing field a little bit, and helps new items get discovered by the masses. So, it plays a very important role in today's society, at least I'd like to think so! There are few ways of doing this, and we'll look at the main ones in this scenario.

User-based collaborative filtering
First, let's talk about recommending stuff based on your past behavior. One technique is called user-based collaborative filtering, and here's how it works:

**Note:**

Collaborative filtering, by the way, is just a fancy name for saying recommending stuff based on the combination of what you did and what everybody else did, okay? So, it's looking at your behavior and comparing that to everyone else's behavior, to arrive at the things that might be interesting to you that you haven't heard of yet.

The idea here is we build up a matrix of everything that every user has ever bought, or viewed, or rated, or whatever signal of interest that you want to base the system on. So basically, we end up with a row for every user in our system, and that row contains all the things they did that might indicate some sort of interest in a given product. So, picture a table, I have users for the rows, and each column is an item, okay? That might be a movie, a product, a web page, whatever; you can use this for many different things.
I then use that matrix to compute the similarity between different users. So, I basically treat each row of this as a vector and I can compute the similarity between each vector of users, based on their behavior.
Two users who liked mostly the same things would be very similar to each other and I can then sort this by those similarity scores. If I can find all the users similar to you based on their past behavior, I can then find the users most similar to me, and recommend stuff that they liked that I didn't look at yet.
Let's look at a real example, and it'll make a little bit more sense:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/3/3.png)

Let's say that this nice lady in the preceding image watched Star Wars and The Empire Strikes Back and she loved them both. So, we have a user vector, of this lady, giving a 5-star rating to Star Wars and The Empire Strikes Back.

Let's also say Mr. Edgy Mohawk Man comes along and he only watched Star Wars. That's the only thing he's seen, he doesn't know about The Empire Strikes Back yet, somehow, he lives in some strange universe where he doesn't know that there are actually many, many Star Wars movies, growing every year in fact.

We can of course say that this guy's actually similar to this other lady because they both enjoyed Star Wars a lot, so their similarity score is probably fairly good and we can say, okay, well, what has this lady enjoyed that he hasn't seen yet? And, The Empire Strikes Back is one, so we can then take that information that these two users are similar based on their enjoyment of Star Wars, find that this lady also liked The Empire Strikes Back, and then present that as a good recommendation for Mr. Edgy Mohawk Man.

We can then go ahead and recommend The Empire Strikes Back to him and he'll probably love it, because in my opinion, it's actually a better film! But I'm not going to get into geek wars with you here.

### Limitations of user-based collaborative filtering

Now, unfortunately, user-based collaborative filtering has some limitations. When we think about relationships and recommending things based on relationships between items and people and whatnot, our mind tends to go on relationships between people. So, we want to find people that are similar to you and recommend stuff that they liked. That's kind of the intuitive thing to do, but it's not the best thing to do! The following is the list of some limitations of user-based collaborative filtering:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/3/4.png)

**Note:**

It's pretty easy to fabricate fake personas in the system by creating a new user and having them do a sequence of events that likes a lot of popular items and then likes your item too. This is called a shilling attack, and we want to ideally have a system that can deal with that.

There is research around how to detect and avoid these shilling attacks in user-based collaborative filtering, but an even better approach would be to use a totally different approach entirely that's not so susceptible to gaming the system.

That's user-based collaborative filtering. Again, it's a simple concept-you look at similarities between users based on their behavior, and recommend stuff that a user enjoyed that was similar to you, that you haven't seen yet. Now, that does have its limitations as we talked about. So, let's talk about flipping the whole thing on its head, with a technique called item-based collaborative filtering.

### Item-based collaborative filtering

Let's now try to address some of the shortcomings in user-based collaborative filtering with a technique called item-based collaborative filtering, and we'll see how that can be more powerful. It's actually one of the techniques that Amazon uses under the hood, and they've talked about this publicly so I can tell you that much, but let's see why it's such a great idea. With user-based collaborative filtering we base our recommendations on relationships between people, but what if we flip that and base them on relationships between items? That's what item-based collaborative filtering is.

**Understanding item-based collaborative filtering**

This is going to draw on a few insights. For one thing, we talked about people being fickle-their tastes can change over time, so comparing one person to another person based on their past behavior becomes pretty complicated. People have different phases where they have different interests, and you might not be comparing the people that are in the same phase to each other. But, an item will always be whatever it is. A movie will always be a movie, it's never going to change. Star Wars will always be Star Wars, well until George Lucas tinkers with it a little bit, but for the most part, items do not change as much as people do. So, we know that these relationships are more permanent, and there's more of a direct comparison you can make when computing similarity between items, because they do not change over time.

The other advantage is that there are generally fewer things that you're trying to recommend than there are people you're recommending to. So again, 7 billion people in the world, you're probably not offering 7 billion things on your website to recommend to them, so you can save a lot of computational resources by evaluating relationships between items instead of users, because you will probably have fewer items than you have users in your system. That means you can run your recommendations more frequently, make them more current, more up-to-date, and better! You can use more complicated algorithms because you have less relationships to compute, and that's a good thing!

It's also harder to game the system. So, we talked about how easy it is to game a user-based collaborative filtering approach by just creating some fake users that like a bunch of popular stuff and then the thing you're trying to promote. With item-based collaborative filtering that becomes much more difficult. You have to game the system into thinking there are relationships between items, and since you probably don't have the capability to create fake items with fake ties to other items based on many, many other users, it's a lot harder to game an item-based collaborative filtering system, which is a good thing.

While I'm on the topic of gaming the system, another important thing is to make sure that people are voting with their money. A general technique for avoiding shilling attacks or people trying to game your recommender system, is to make sure that the signal behavior is based on people actually spending money. So, you're always going to get better and more reliable results when you base recommendations on what people actually bought, as opposed to what they viewed or what they clicked on, okay?

### How item-based collaborative filtering works?

Alright, let's talk about how item-based collaborative filtering works. It's very similar to user-based collaborative filtering, but instead of users, we're looking at items.

So, let's go back to the example of movie recommendations. The first thing we would do is find every pair of movies that is watched by the same person. So, we go through and find every movie that was watched by identical people, and then we measure the similarity of all those people who viewed that movie to each other. So, by this means we can compute similarities between two different movies, based on the ratings of the people who watched both of those movies.

So, let's presume I have a movie pair, okay? Maybe Star Wars and The Empire Strikes Back. I find a list of everyone who watched both of those movies, then I compare their ratings to each other, and if they're similar then I can say these two movies are similar, because they were rated similarly by people who watched both of them. That's the general idea here. That's one way to do it, there's more than one way to do it!

And then I can just sort everything by the movie, and then by the similarity strength of all the similar movies to it, and there's my results for people who liked also liked, or people who rated this highly also rated this highly and so on and so forth. And like I said, that's just one way of doing it.


That's step one of item-based collaborative filtering-first I find relationships between movies based on the relationships of the people who watched every given pair of movies. It'll make more sense when we go through the following example:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/8/1.png)

For example, let's say that our nice young lady in the preceding image watched Star Wars and The Empire Strikes Back and liked both of them, so rated them both five stars or something. Now, along comes Mr. Edgy Mohawk Man who also watched Star Wars and The Empire Strikes Back and also liked both of them. So, at this point we can say there's a relationship, there is a similarity between Star Wars and The Empire Strikes Back based on these two users who liked both movies.

What we're going to do is look at each pair of movies. We have a pair of Star Wars and Empire Strikes Back, and then we look at all the users that watched both of them, which are these two guys, and if they both liked them, then we can say that they're similar to each other. Or, if they both disliked them we can also say they're similar to each other, right? So, we're just looking at the similarity score of these two users' behavior related to these two movies in this movie pair.

So, along comes Mr. Moustachy Lumberjack Hipster Man and he watches The Empire Strikes Back and he lives in some strange world where he watched The Empire Strikes Back, but had no idea that Star Wars the first movie existed.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/8/2.png)

Well that's fine, we computed a relationship between The Empire Strikes Back and Star Wars based on the behavior of these two people, so we know that these two movies are similar to each other. So, given that Mr. Hipster Man liked The Empire Strikes Back, we can say with good confidence that he would also like Star Wars, and we can then recommend that back to him as his top movie recommendation. Something like the following illustration:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/8/3.png)

You can see that you end up with very similar results in the end, but we've kind of flipped the whole thing on its head. So, instead of focusing the system on relationships between people, we're focusing them on relationships between items, and those relationships are still based on the aggregate behavior of all the people that watch them. But fundamentally, we're looking at relationships between items and not relationships between people. Got it?

### Collaborative filtering using Python

Alright, so let's do it! We have some Python code that will use Pandas, and all the various other tools at our disposal, to create movie recommendations with a surprisingly little amount of code.

The first thing we're going to do is show you item-based collaborative filtering in practice. So, we'll build up people who watched also watched basically, you know, people who rated things highly also rated this thing highly, so building up these movie to movie relationships. So, we're going to base it on real data that we got from the MovieLens project. So, if you go to MovieLens.org, there's actually an open movie recommender system there, where people can rate movies and get recommendations for new movies.

And, they make all the underlying data publicly available for researchers like us. So, we're going to use some real movie ratings data-it is a little bit dated, it's like 10 years old, so keep that in mind, but it is real behavior data that we're going to be working with finally here. And, we will use that to compute similarities between movies. And, that data in and of itself is useful. You can use that data to say people who liked also liked. So, let's say I'm looking at a web page for a movie. the system can then say: if you liked this movie, and given that you're looking at it you're probably interested in it, then you might also like these movies. And that's a form of a recommender system right there, even though we don't even know who you are.

Now, it is real-world data, so we're going to encounter some real-world problems with it. Our initial set of results aren't going to look good, so we're going to spend a little bit of extra time trying to figure out why, which is a lot of what you spend your time doing as a data scientist-correct those problems, and go back and run it again until we get results that makes sense.

And finally, we'll actually do item-based collaborative filtering in its entirety, where we actually recommend movies to individuals based on their own behavior. So, let's do this,let's get started!

### Finding movie similarities

Let's apply the concept of item-based collaborative filtering. To start with, movie similarities-figure out what movies are similar to other movies. In particular, we'll try to figure out what movies are similar to Star Wars, based on user rating data, and we'll see what we get out of it. Let's dive in!

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `SimilarMovies.ipynb` in the `work` folder.



Okay so, let's go ahead and compute the first half of item-based collaborative filtering, which is finding similarities between items.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/11/1.png)

In this case, we're going to be looking at similarities between movies, based on user behavior. And, we're going to be using some real movie rating data from the GroupLens project. GroupLens.org provides real movie ratings data, by real people who are using the MovieLens.org website to rate movies and get recommendations back for new movies that they want to watch.


We have included the data files that you need from the GroupLens dataset with the course materials, and the first thing we need to do is import those into a Pandas DataFrame, and we're really going to see the full power of Pandas in this example. It's pretty cool stuff!

Understanding the code
The first thing we're going to do is import the u.data file as part of the MovieLens dataset, and that is a tab-delimited file that contains every rating in the dataset.

```
import pandas as pd 
 
r_cols = ['user_id', 'movie_id', 'rating'] 
ratings = pd.read_csv('./ml-100k/u.data',  
                      sep='\\t', names=r_cols, usecols=range(3)) 
```
                      
**Note:** that you'll need to add the path here to where you stored the downloaded MovieLens files on your computer. So, the way that this works is even though we're calling read_csv on Pandas, we can specify a different separator than a comma. In this case, it's a tab.

We're basically saying take the first three columns in the u.data file, and import it into a new DataFrame, with three columns: user_id, movie_id, and rating.

What we end up with here is a DataFrame that has a row for every user_id, which identifies some person, and then, for every movie they rated, we have the movie_id, which is some numerical shorthand for a given movie, so Star Wars might be movie 53 or something, and their rating, you know, 1 to 5 stars. So, we have here a database, a DataFrame, of every user and every movie they rated, okay?

Now, we want to be able to work with movie titles, so we can interpret these results more intuitively, so we're going to use their human-readable names instead.

If you're using a truly massive dataset, you'd save that to the end because you want to be working with numbers, they're more compact, for as long as possible. For the purpose of example and teaching, though, we'll keep the titles around so you can see what's going on.

```
m_cols = ['movie_id', 'title'] 
movies = pd.read_csv('./ml-100k/u.item', 
                     sep='|', names=m_cols, usecols=range(2)) 
```

There's a separate data file with the MovieLens dataset called u.item, and it is pipe-delimited, and the first two columns that we import will be the movie_id and the title of that movie. So, now we have two DataFrames: r_cols has all the user ratings and m_cols has all the titles for every movie_id. We can then use the magical merge function in Pandas to mush it all together.

```
ratings = pd.merge(movies, ratings) 
```

Let's add a ratings.head() command and then run those cells. What we end up with is something like the following table. That was pretty quick!

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/11/2.png)

We end up with a new DataFrame that contains the user_id and rating for each movie that a user rated, and we have both the movie_id and the title that we can read and see what it really is. So, the way to read this is user_id number 308 rated the Toy Story (1995) movie 4 stars, user_id number 287 rated the Toy Story (1995) movie 5 stars, and so on and so forth. And, if we were to keep looking at more and more of this DataFrame, we'd see different ratings for different movies as we go through it.


Now the real magic of Pandas comes in. So, what we really want is to look at relationships between movies based on all the users that watched each pair of movies, so we need, at the end, a matrix of every movie, and every user, and all the ratings that every user gave to every movie. The pivot_table command in Pandas can do that for us. It can basically construct a new table from a given DataFrame, pretty much any way that you want it. For this, we can use the following code:

```
movieRatings = ratings.pivot_table(index=['user_id'],
                                   columns=['title'],values='rating') 
movieRatings.head() 
```

So, what we're saying with this code is-take our ratings DataFrame and create a new DataFrame called movieRatings and we want the index of it to be the user IDs, so we'll have a row for every user_id, and we're going to have every column be the movie title. So, we're going to have a column for every title that we encounter in that DataFrame, and each cell will contain the rating value, if it exists. So, let's go ahead and run it.

And, we end up with a new DataFrame that looks like the following table:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/11/3.png)

It's kind of amazing how that just put it all together for us. Now, you'll see some NaN values, which stands for Not a Number, and its just how Pandas indicates a missing value. So, the way to interpret this is, user_id number 1, for example, did not watch the movie 1-900 (1994), but user_id number 1 did watch 101 Dalmatians (1996) and rated it 2 stars. The user_id number 1 also watched 12 Angry Men (1957) and rated it 5 stars, but did not watch the movie 2 Days in the Valley (1996), for example, okay? So, what we end up with here is a sparse matrix basically, that contains every user, and every movie, and at every intersection where a user rated a movie there's a rating value.

So, you can see now, we can very easily extract vectors of every movie that our user watched, and we can also extract vectors of every user that rated a given movie, which is what we want. So, that's useful for both user-based and item-based collaborative filtering, right? If I wanted to find relationships between users, I could look at correlations between these user rows, but if I want to find correlations between movies, for item-based collaborative filtering, I can look at correlations between columns based on the user behavior. So, this is where the real flipping things on its head for user versus item-based similarities comes into play.

Now, we're going with item-based collaborative filtering, so we want to extract columns, to do this let's run the following code:

```
starWarsRatings = movieRatings['Star Wars (1977)'] 
starWarsRatings.head() 
```

Now, with the help of that, let's go ahead and extract all the users who rated Star Wars (1977):

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/11/4.png)

And, we can see most people have, in fact, watched and rated Star Wars (1977) and everyone liked it, at least in this little sample that we took from the head of the DataFrame. So, we end up with a resulting set of user IDs and their ratings for Star Wars (1977). The user ID 3 did not rate Star Wars (1977) so we have a NaN value, indicating a missing value there, but that's okay. We want to make sure that we preserve those missing values so we can directly compare columns from different movies. So, how do we do that?

The corrwith function
Well, Pandas keeps making it easy for us, and has a corrwith function that you can see in the following code that we can use:

```
similarMovies = movieRatings.corrwith(starWarsRatings) 
similarMovies = similarMovies.dropna() 
df = pd.DataFrame(similarMovies) 
df.head(10) 
```

That code will go ahead and correlate a given column with every other column in the DataFrame, and compute the correlation scores and give that back to us. So, what we're doing here is using corrwith on the entire movieRatings DataFrame, that's that entire matrix of user movie ratings, correlating it with just the starWarsRatings column, and then dropping all of the missing results with dropna. So, that just leaves us with items that had a correlation, where there was more than one person that viewed it, and we create a new DataFrame based on those results and then display the top 10 results. So again, just to recap:

We're going to build the correlation score between Star Wars and every other movie.
Drop all the NaN values, so that we only have movie similarities that actually exist, where more than one person rated it.
And, we're going to construct a new DataFrame from the results and look at the top 10 results.
And here we are with the results shown in the following screenshot:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/11/5.png)

We ended up with this result of correlation scores between each individual movie for Star Wars and we can see, for example, a surprisingly high correlation score with the movie 'Til There Was You (1997), a negative correlation with the movie 1-900 (1994), and a very weak correlation with 101 Dalmatians (1996).

Now, all we should have to do is sort this by similarity score, and we should have the top movie similarities for Star Wars, right? Let's go ahead and do that.

```
similarMovies.sort_values(ascending=False)
```

Just call sort_values on the resulting DataFrame, again Pandas makes it really easy, and we can say ascending=False, to actually get it sorted in reverse order by correlation score. So, let's do that:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/11/6.png)

Okay, so Star Wars (1977) came out pretty close to top, because it is similar to itself, but what's all this other stuff? What the heck? We can see in the preceding output, some movies such as: Full Speed (1996), Man of the Year (1995), The Outlaw (1943). These are all, you know, fairly obscure movies, that most of them I've never even heard of, and yet they have perfect correlations with Star Wars. That's kinda weird! So, obviously we're doing something wrong here. What could it be?

Well, it turns out there's a perfectly reasonable explanation, and this is a good lesson in why you always need to examine your results when you're done with any sort of data science task-question the results, because often there's something you missed, there might be something you need to clean in your data, there might be something you did wrong. But you should also always look skeptically at your results, don't just take them on faith, okay? If you do so, you're going to get in trouble, because if I were to actually present these as recommendations to people who liked Star Wars, I would get fired. Don't get fired! Pay attention to your results! So, let's dive into what went wrong in our next section.

### Improving the results of movie similarities

Let's figure out what went wrong with our movie similarities there. We went through all this exciting work to compute correlation scores between movies based on their user ratings vectors, and the results we got kind of sucked. So, just to remind you, we looked for movies that are similar to Star Wars using that technique, and we ended up with a bunch of weird recommendations at the top that had a perfect correlation.

And, most of them were very obscure movies. So, what do you think might be going on there? Well, one thing that might make sense is, let's say we have a lot of people watch Star Wars and some other obscure film. We'd end up with a good correlation between these two movies because they're tied together by Star Wars, but at the end of the day, do we really want to base our recommendations on the behavior of one or two people that watch some obscure movie?

Probably not! I mean yes, the two people in the world, or whatever it is, that watch the movie Full Speed, and both liked it in addition to Star Wars, maybe that is a good recommendation for them, but it's probably not a good recommendation for the rest of the world. We need to have some sort of confidence level in our similarities by enforcing a minimum boundary of how many people watched a given movie. We can't make a judgment that a given movie is good just based on the behavior of one or two people.

So, let's try to put that insight into action using the following code:

```
import numpy as np 
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]}) 
movieStats.head() 
```

What we're going to do is try to identify the movies that weren't actually rated by many people and we'll just throw them out and see what we get. So, to do that we're going to take our original ratings DataFrame and we're going to say groupby('title'), again Pandas has all sorts of magic in it. And, this will basically construct a new DataFrame that aggregates together all the rows for a given title into one row.


We can say that we want to aggregate specifically on the rating, and we want to show both the size, the number of ratings for each movie, and the mean average score, the mean rating for that movie. So, when we do that, we end up with something like the following:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/14/1.jpg)

This is telling us, for example, for the movie 101 Dalmatians (1996), 109 people rated that movie and their average rating was 2.9 stars, so not that great of a score really! So, if we just eyeball this data, we can say okay well, movies that I consider obscure, like 187 (1997), had 41 ratings, but 101 Dalmatians (1996), I've heard of that, you know 12 Angry Men (1957), I've heard of that. It seems like there's sort of a natural cutoff value at around 100 ratings, where maybe that's the magic value where things start to make sense.

Let's go ahead and get rid of movies rated by fewer than 100 people, and yes, you know I'm kind of doing this intuitively at this point. As we'll talk about later, there are more principled ways of doing this, where you could actually experiment and do train/test experiments on different threshold values, to find the one that actually performs the best. But initially, let's just use our common sense and filter out movies that were rated by fewer than 100 people. Again, Pandas makes that really easy to do. Let's figure it out with the following example:

```
popularMovies = movieStats['rating']['size'] >= 100 
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15] 
```

We can just say popularMovies, a new DataFrame, is going to be constructed by looking at movieStats and we're going to only take rows where the rating size is greater than or equal to 100, and I'm then going to sort that by mean rating, just for fun, to see the top rated, widely watched movies.

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/14/2.jpg)

What we have here is a list of movies that were rated by more than 100 people, sorted by their average rating score, and this in itself is a recommender system. These are highly-rated popular movies. A Close Shave (1995), apparently, was a really good movie and a lot of people watched it and they really liked it.

So again, this is a very old dataset, from the late 90s, so even though you might not be familiar with the film A Close Shave (1995), it might be worth going back and rediscovering it; add it to your Netflix! Schindler's List (1993) not a big surprise there,that comes up on the top of most top movies lists. The Wrong Trousers (1993), another example of an obscure film that apparently was really good and was also pretty popular. So,some interesting discoveries there already, just by doing that.


Things look a little bit better now, so let's go ahead and basically make our new DataFrame of Star Wars recommendations, movies similar to Star Wars, where we only base it on movies that appear in this new DataFrame. So, we're going to use the join operation, to go ahead and join our original similarMovies DataFrame to this new DataFrame of only movies that have greater than 100 ratings, okay?

```
df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity'])) 
df.head() 
```

In this code, we create a new DataFrame based on similarMovies where we extract the similarity column, join that with our movieStats DataFrame, which is our popularMovies DataFrame, and we look at the combined results. And, there we go with that output!

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/14/3.jpg)

Now we have, restricted only to movies that are rated by more than 100 people, the similarity score to Star Wars. So, now all we need to do is sort that using the following code:

```
df.sort_values(['similarity'], ascending=False)[:15] 
```

Here, we're reverse sorting it and we're just going to take a look at the first 15 results. If you run that now, you should see the following:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/14/4.jpg)

This is starting to look a little bit better! So, Star Wars (1977) comes out on top because it's similar to itself, The Empire Strikes Back (1980) is number 2, Return of the Jedi (1983) is number 3, Raiders of the Lost Ark (1981), number 4. You know, it's still not perfect, but these make a lot more sense, right? So, you would expect the three Star Wars films from the original trilogy to be similar to each other, this data goes back to before the next three films, and Raiders of the Lost Ark (1981) is also a very similar movie to Star Wars in style, and it comes out as number 4. So, I'm starting to feel a little bit better about these results. There's still room for improvement, but hey! We got some results that make sense, whoo-hoo!

Now, ideally, we'd also filter out Star Wars, you don't want to be looking at similarities to the movie itself that you started from, but we'll worry about that later! So, if you want to play with this a little bit more, like I said 100 was sort of an arbitrary cutoff for the minimum number of ratings. If you do want to experiment with different cutoff values, I encourage you to go back and do so. See what that does to the results. You know, you can see in the preceding table that the results that we really like actually had much more than 100 ratings in common. So, we end up with Austin Powers: International Man of Mystery (1997) coming in there pretty high with only 130 ratings so maybe 100 isn't high enough! Pinocchio (1940) snuck in at 101, not very similar to Star Wars, so, you might want to consider an even higher threshold there and see what it does.

**Note:**

Please keep in mind too, this is a very small, limited dataset that we used for experimentation purposes, and it's based on very old data, so you're only going to see older movies. So, interpreting these results intuitively might be a little bit challenging as a result, but not bad results.

Now let's move on and actually do full-blown item-based collaborative filtering where we recommend movies to people using a more complete system, we'll do that next.

### Making movie recommendations to people

Okay, let's actually build a full-blown recommender system that can look at all the behavior information of everybody in the system, and what movies they rated, and use that to actually produce the best recommendation movies for any given user in our dataset. Kind of amazing and you'll be surprised how simple it is. Let's go!

#### Open Notebook
The Notebook opens in a new browser window. You can create a new notebook or open a local one. Check out the local folder `work` for several notebooks. Open and run `ItemBasedCF.ipynb` in the `work` folder.




Let's begin using the `ItemBasedCF.ipynb` file and let's start off by importing the MovieLens dataset that we have. Again, we're using a subset of it that just contains 100,000 ratings for now. But, there are larger datasets you can get from GroupLens.org-up to millions of ratings; if you're so inclined. Keep in mind though, when you start to deal with that really big data, you're going to be pushing the limits of what you can do in a single machine and what Pandas can handle. Without further ado, here's the first block of code:

```
import pandas as pd 
 
r_cols = ['user_id', 'movie_id', 'rating'] 
ratings = pd.read_csv('./ml-100k/u.data',      
                      sep='\t', names=r_cols, usecols=range(3)) 
 
m_cols = ['movie_id', 'title'] 
movies = pd.read_csv('./ml-100k/u.item', 
                     sep='|', names=m_cols, usecols=range(2)) 
 
ratings = pd.merge(movies, ratings) 
 
ratings.head() 
```

Just like earlier, we're going to import the u.data file that contains all the individual ratings for every user and what movie they rated, and then we're going to tie that together with the movie titles, so we don't have to just work with numerical movie IDs. Go ahead and hit the run cell button, and we end up with the following DataFrame.


![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/17/1.jpg)

The way to read this is, for example, user_id number 308 rated Toy Story (1995) a 4 star, and user_id number 66 rated Toy Story (1995) a 3 star. And, this will contain every rating, for every user, for every movie.


And again, just like earlier, we use the wonderful pivot_table command in Pandas to construct a new DataFrame based on the information:

```
userRatings = ratings.pivot_table(index=['user_id'],
                                  columns=['title'],values='rating') 
userRatings.head() 
```

Here, each row is the user_id, the columns are made up of all the unique movie titles in my dataset, and each cell contains a rating:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/17/2.jpg)

What we end up with is this incredibly useful matrix shown in the preceding output, that contains users for every row and movies for every column. And we have basically every user rating for every movie in this matrix. So, user_id number 1, for example, gave 101 Dalmatians (1996) a 2-star rating. And, again all these NaN values represent missing data. So, that just indicates, for example, user_id number 1 did not rate the movie 1-900 (1994).

Again, it's a very useful matrix to have. If we were doing user-based collaborative filtering, we could compute correlations between each individual user rating vector to find similar users. Since we're doing item-based collaborative filtering, we're more interested in relationships between the columns. So, for example, doing a correlation score between any two columns, which will give us a correlation score for a given movie pair. So, how do we do that? It turns out that Pandas makes that incredibly easy to do as well.

It has a built-in corr function that will actually compute the correlation score for every column pair found in the entire matrix-it's almost like they were thinking of us.

```
corrMatrix = userRatings.corr() 
corrMatrix.head() 
```

Let's go ahead and run the preceding code. It's a fairly computationally expensive thing to do, so it will take a moment to actually come back with a result. But, there we have it!

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/17/3.jpg)

So, what do we have in the preceding output? We have here a new DataFrame where every movie is on the row, and in the column. So, we can look at the intersection of any two given movies and find their correlation score to each other based on this userRatings data that we had up here originally. How cool is that? For example, the movie 101 Dalmatians (1996) is perfectly correlated with itself of course, because it has identical user rating vectors. But, if you look at 101 Dalmatians (1996) movie's relationship to the movie 12 Angry Men (1957), it's a much lower correlation score because those movies are rather dissimilar, makes sense, right?

I have this wonderful matrix now that will give me the similarity score of any two movies to each other. It's kind of amazing, and very useful for what we're going to be doing. Now just like earlier, we have to deal with spurious results. So, I don't want to be looking at relationships that are based on a small amount of behavior information.

It turns out that the Pandas corr function actually has a few parameters you can give it. One is the actual correlation score method that you want to use, so I'm going to say use pearson correlation.

```
corrMatrix = userRatings.corr(method='pearson', min_periods=100) 
corrMatrix.head() 
```

You'll notice that it also has a min_periods parameter you can give it, and that basically says I only want you to consider correlation scores that are backed up by at least, in this example, 100 people that rated both movies. Running that will get rid of the spurious relationships that are based on just a handful of people. The following is the matrix that we get after running the code:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/17/4.jpg)

It's a little bit different to what we did in the item similarities exercise where we just threw out any movie that was rated by less than 100 people. What we're doing here, is throwing out movie similarities where less than 100 people rated both of those movies, okay? So, you can see in the preceding matrix that we have a lot more NaN values.

In fact, even movies that are similar to themselves get thrown out, so for example, the movie 1-900 (1994) was, presumably, watched by fewer than 100 people so it just gets tossed entirely. The movie, 101 Dalmatians (1996) however, survives with a correlation score of 1, and there are actually no movies in this little sample of the dataset that are different from each other that had 100 people in common that watched both. But, there are enough movies that survive to get meaningful results.

### Understanding movie recommendations with an example

So, what we do with this data? Well, what we want to do is recommend movies for people. The way we do that is we look at all the ratings for a given person, find movies similar to the stuff that they rated, and those are candidates for recommendations to that person.

Let's start by creating a fake person to create recommendations for. I've actually already added a fake user by hand, ID number 0, to the MovieLens dataset that we're processing. You can see that user with the following code:

```
myRatings = userRatings.loc[0].dropna() 
myRatings 
```

This gives the following output:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/17/5.jpg)


That kind of represents someone like me, who loved Star Wars and The Empire Strikes Back, but hated the movie Gone with the Wind. So, this represents someone who really loves Star Wars, but does not like old style, romantic dramas, okay? So, I gave a rating of 5 star to The Empire Strikes Back (1980) and Star Wars (1977), and a rating of 1 star to Gone with the Wind (1939). So, I'm going to try to find recommendations for this fictitious user.


So, how do I do that? Well, let's start by creating a series called simCandidates and I'm going to go through every movie that I rated.

```
simCandidates = pd.Series() 
for i in range(0, len(myRatings.index)): 
    print "Adding sims for " + myRatings.index[i] + "..." 
    # Retrieve similar movies to this one that I rated 
    sims = corrMatrix[myRatings.index[i]].dropna() 
    # Now scale its similarity by how well I rated this movie 
    sims = sims.map(lambda x: x * myRatings[i]) 
    # Add the score to the list of similarity candidates 
    simCandidates = simCandidates.append(sims) 
     
#Glance at our results so far: 
print "sorting..." 
simCandidates.sort_values(inplace = True, ascending = False) 
print simCandidates.head(10) 
```

For i in range 0 through the number of ratings that I have in myRatings, I am going to add up similar movies to the ones that I rated. So, I'm going to take that corrMatrix DataFrame, that magical one that has all of the movie similarities, and I am going to create a correlation matrix with myRatings, drop any missing values, and then I am going to scale that resulting correlation score by how well I rated that movie.

So, the idea here is I'm going to go through all the similarities for The Empire Strikes Back, for example, and I will scale it all by 5, because I really liked The Empire Strikes Back. But, when I go through and get the similarities for Gone with the Wind, I'm only going to scale those by 1, because I did not like Gone with the Wind. So, this will give more strength to movies that are similar to movies that I liked, and less strength to movies that are similar to movies that I did not like, okay?


So, I just go through and build up this list of similarity candidates, recommendation candidates if you will, sort the results and print them out. Let's see what we get:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/17/6.jpg)

Hey, those don't look too bad, right? So, obviously The Empire Strikes Back (1980) and Star Wars (1977) come out on top, because I like those movies explicitly, I already watched them and rated them. But, bubbling up to the top of the list is Return of the Jedi (1983), which we would expect and Raiders of the Lost Ark (1981).

Let's start to refine these results a little bit more. We're seeing that we're getting duplicate values back. If we have a movie that was similar to more than one movie that I rated, it will come back more than once in the results, so we want to combine those together. If I do in fact have the same movie, maybe that should get added up together into a combined, stronger recommendation score. Return of the Jedi, for example, was similar to both Star Wars and The Empire Strikes Back. How would we do that?

Using the groupby command to combine rows
We'll go ahead and explore that. We're going to use the groupby command again to group together all of the rows that are for the same movie. Next, we will sum up their correlation score and look at the results:

```
simCandidates = simCandidates.groupby(simCandidates.index).sum() 
simCandidates.sort_values(inplace = True, ascending = False) 
simCandidates.head(10) 
```

Following is the result:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/17/7.jpg)

Hey, this is looking really good!

So Return of the Jedi (1983) comes out way on top, as it should, with a score of 7, Raiders of the Lost Ark (1981) a close second at 5, and then we start to get to Indiana Jones and the Last Crusade (1989), and some more movies, The Bridge on the River Kwai (1957), Back to the Future (1985),The Sting (1973). These are all movies that I would actually enjoy watching! You know, I actually do like old-school Disney movies too, so Cinderella (1950) isn't as crazy as it might seem.

The last thing we need to do is filter out the movies that I've already rated, because it doesn't make sense to recommend movies you've already seen.

Removing entries with the drop command
So, I can quickly drop any rows that happen to be in my original ratings series using the following code:

```
filteredSims = simCandidates.drop(myRatings.index) 
filteredSims.head(10) 
```

Running that will let me see the final top 10 results:

![](https://github.com/fenago/datascience-machine-learning/raw/master/images/datascience-machine-learning-chapter-06/steps/17/8.jpg)

And there we have it! Return of the Jedi (1983), Raiders of the Lost Ark (1981), Indiana Jones and the Last Crusade (1989), all the top results for my fictitious user, and they all make sense. I'm seeing a few family-friendly films, you know, Cinderella (1950), The Wizard of Oz (1939), Dumbo (1941), creeping in, probably based on the presence of Gone with the Wind in there, even though it was weighted downward it's still in there, and still being counted. And, there we have our results, so. There you have it! Pretty cool!

We have actually generated recommendations for a given user and we could do that for any user in our entire DataFrame. So, go ahead and play with that if you want to. I also want to talk about how you can actually get your hands dirty a little bit more, and play with these results; try to improve upon them.

There's a bit of an art to this, you know, you need to keep iterating and trying different ideas and different techniques until you get better and better results, and you can do this pretty much forever. I mean, I made a whole career out of it. So, I don't expect you to spend the next, you know, 10 years trying to refine this like I did, but there are some simple things you can do, so let's talk about that.

### Improving the recommendation results

As an exercise, I want to challenge you to go and make those recommendations even better. So, let's talk about some ideas I have, and maybe you'll have some of your own too that you can actually try out and experiment with; get your hands dirty, and try to make better movie recommendations.

Okay, there's a lot of room for improvement still on these recommendation results. There's a lot of decisions we made about how to weigh different recommendation results based on your rating of that item that it came from, or what threshold you want to pick for the minimum number of people that rated two given movies. So, there's a lot of things you can tweak, a lot of different algorithms you can try, and you can have a lot of fun with trying to make better movie recommendations out of the system. So, if you're feeling up to it, I'm challenging you to go and do just that!

Here are some ideas on how you might actually try to improve upon the results in this scenario. First, you can just go ahead and play with the `ItembasedCF.ipynb` file and tinker with it. So, for example, we saw that the correlation method actually had some parameters for the correlation computation, we used Pearson in our example, but there are other ones you can look up and try out, see what it does to your results. We used a minimum period value of 100, maybe that's too high, maybe it's too low; we just kind of picked it arbitrarily. What happens if you play with that value? If you were to lower that for example, I would expect you to see some new movies maybe you've never heard of, but might still be a good recommendation for that person. Or, if you were to raise it higher, you would see, you know nothing but blockbusters.

Sometimes you have to think about what the result is that you want out of a recommender system. Is there a good balance to be had between showing people movies that they've heard of and movies that they haven't heard of? How important is discovery of new movies to these people versus having confidence in the recommender system by seeing a lot of movies that they have heard of? So again, there's sort of an art to that.

We can also improve upon the fact that we saw a lot of movies in the results that were similar to Gone with the Wind, even though I didn't like Gone with the Wind. You know we weighted those results lower than similarities to movies that I enjoyed, but maybe those movies should actually be penalized. If I hated Gone with the Wind that much, maybe similarities to Gone with the Wind, like The Wizard of Oz, should actually be penalized and, you know lowered in their score instead of raised at all.

That's another simple modification you can make and play around with. There are probably some outliers in our user rating dataset, what if I were to throw away people that rated some ridiculous number of movies? Maybe they're skewing everything. You could actually try to identify those users and throw them out, as another idea. And, if you really want a big project, if you really want to sink your teeth into this stuff, you could actually evaluate the results of this recommender engine by using the techniques of train/test. So, what if instead of having an arbitrary recommendation score that sums up the correlation scores of each individual movie, actually scale that down to a predicted rating for each given movie.


If the output of my recommender system were a movie and my predicted rating for that movie, in a train/test system I could actually try to figure out how well do I predict movies that the user has in fact watched and rated before? Okay? So, I could set aside some of the ratings data and see how well my recommender system is able to predict the user's ratings for those movies. And, that would be a quantitative and principled way to measure the error of this recommender engine. But again, there's a little bit more of an art than a science to this. Even though the Netflix prize actually used that error metric, called root-mean-square error is what they used in particular, is that really a measure of a good recommender system?

Basically, you're measuring the ability of your recommender system to predict the ratings of movies that a person already watched. But isn't the purpose of a recommender engine to recommend movies that a person hasn't watched, that they might enjoy? Those are two different things. So unfortunately, it's not very easy to measure the thing you really want to be measuring. So sometimes, you do kind of have to go with your gut instinct. And, the right way to measure the results of a recommender engine is to measure the results that you're trying to promote through it.

Maybe I'm trying to get people to watch more movies, or rate new movies more highly, or buy more stuff. Running actual controlled experiments on a real website would be the right way to optimize for that, as opposed to using train/test. So, you know, I went into a little bit more detail there than I probably should have, but the lesson is, you can't always think about these things in black and white. Sometimes, you can't really measure things directly and quantitatively, and you have to use a little bit of common sense, and this is an example of that.

Anyway, those are some ideas on how to go back and improve upon the results of this recommender engine that we wrote. So, please feel free to tinker around with it, see if you can improve upon it however you wish to, and have some fun with it. This is actually a very interesting part of the book, so I hope you enjoy it!

