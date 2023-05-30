# Data Cleaning in Pandas - Introduction

## Introduction

In this section, you will learn invaluable skills that will form the foundation of your data processing work. Before you can apply machine learning algorithms or do interesting analyses, you often must clean and transform your data into a suitable format. Such initial data wrangling processes are often referred to as Extract Transform Load (ETL). Our primary tool of choice for performing ETL and basic analyses will be the Pandas package.



## Why ETL?

ETL is an essential first step to data analysis and data science. It also will form the foundation for exploratory data analysis. Often, you will be thrown a dataset that you have little to no information about. In these cases, your first step is to explore the data and get familiar with it. What are the columns? How many observations do you have? Are there missing values? Any outliers? If we have user-level data, how can we explore aggregate trends along features like gender, race, or geography? All of these can be answered by applying ETL to transform raw datasets into alternative useful views. 

## Quick ETL Example

While you'll see complete examples and explanations for all of these techniques (and more), here's a quick preview of some ETL techniques covered in this section! For more details, continue on to future lessons!

### Import data


```python
import pandas as pd
df = pd.read_csv('Yelp_Reviews.csv', index_col=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>cool</th>
      <th>date</th>
      <th>funny</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>useful</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>pomGBqfbxcqPv14c3XH-ZQ</td>
      <td>0</td>
      <td>2012-11-13</td>
      <td>0</td>
      <td>dDl8zu1vWPdKGihJrwQbpw</td>
      <td>5</td>
      <td>I love this place! My fiance And I go here atl...</td>
      <td>0</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>jtQARsP6P-LbkyjbO1qNGg</td>
      <td>1</td>
      <td>2014-10-23</td>
      <td>1</td>
      <td>LZp4UX5zK3e-c5ZGSeo3kA</td>
      <td>1</td>
      <td>Terrible. Dry corn bread. Rib tips were all fa...</td>
      <td>3</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ums3gaP2qM3W1XcA5r6SsQ</td>
      <td>0</td>
      <td>2014-09-05</td>
      <td>0</td>
      <td>jsDu6QEJHbwP2Blom1PLCA</td>
      <td>5</td>
      <td>Delicious healthy food. The steak is amazing. ...</td>
      <td>0</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
    </tr>
    <tr>
      <th>5</th>
      <td>vgfcTvK81oD4r50NMjU2Ag</td>
      <td>0</td>
      <td>2011-02-25</td>
      <td>0</td>
      <td>pfavA0hr3nyqO61oupj-lA</td>
      <td>1</td>
      <td>This place sucks. The customer service is horr...</td>
      <td>2</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
    </tr>
    <tr>
      <th>10</th>
      <td>yFumR3CWzpfvTH2FCthvVw</td>
      <td>0</td>
      <td>2016-06-15</td>
      <td>0</td>
      <td>STiFMww2z31siPY7BWNC2g</td>
      <td>5</td>
      <td>I have been an Emerald Club member for a numbe...</td>
      <td>0</td>
      <td>TlvV-xJhmh7LCwJYXkV-cg</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (2610, 9)



### Apply lambda functions


```python
df['Review_Word_Length'] = df['text'].map(lambda x: len(x.split()))
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>cool</th>
      <th>date</th>
      <th>funny</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>useful</th>
      <th>user_id</th>
      <th>Review_Word_Length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>pomGBqfbxcqPv14c3XH-ZQ</td>
      <td>0</td>
      <td>2012-11-13</td>
      <td>0</td>
      <td>dDl8zu1vWPdKGihJrwQbpw</td>
      <td>5</td>
      <td>I love this place! My fiance And I go here atl...</td>
      <td>0</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
      <td>58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>jtQARsP6P-LbkyjbO1qNGg</td>
      <td>1</td>
      <td>2014-10-23</td>
      <td>1</td>
      <td>LZp4UX5zK3e-c5ZGSeo3kA</td>
      <td>1</td>
      <td>Terrible. Dry corn bread. Rib tips were all fa...</td>
      <td>3</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ums3gaP2qM3W1XcA5r6SsQ</td>
      <td>0</td>
      <td>2014-09-05</td>
      <td>0</td>
      <td>jsDu6QEJHbwP2Blom1PLCA</td>
      <td>5</td>
      <td>Delicious healthy food. The steak is amazing. ...</td>
      <td>0</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>vgfcTvK81oD4r50NMjU2Ag</td>
      <td>0</td>
      <td>2011-02-25</td>
      <td>0</td>
      <td>pfavA0hr3nyqO61oupj-lA</td>
      <td>1</td>
      <td>This place sucks. The customer service is horr...</td>
      <td>2</td>
      <td>msQe1u7Z_XuqjGoqhB0J5g</td>
      <td>82</td>
    </tr>
    <tr>
      <th>10</th>
      <td>yFumR3CWzpfvTH2FCthvVw</td>
      <td>0</td>
      <td>2016-06-15</td>
      <td>0</td>
      <td>STiFMww2z31siPY7BWNC2g</td>
      <td>5</td>
      <td>I have been an Emerald Club member for a numbe...</td>
      <td>0</td>
      <td>TlvV-xJhmh7LCwJYXkV-cg</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape # Previously this was (2610, 9), now we have added a column
```




    (2610, 10)



### Group data


```python
df.groupby('business_id')['stars'].mean().head()
```




    business_id
    -050d_XIor1NpCuWkbIVaQ    5.0
    -0qht1roIqleKiQkBLDkbw    1.0
    -3zffZUHoY8bQjGfPSoBKQ    5.0
    -6tvduBzjLI1ISfs3F_qTg    5.0
    -9nai28tnoylwViuJVrYEQ    5.0
    Name: stars, dtype: float64



### Check for duplicates

Check how many we have:


```python
df.duplicated().value_counts()
```




    False    2277
    True      333
    dtype: int64



Visually inspect them:


```python
# Use keep=False to keep all duplicates and sort_values to put duplicates next to each other
df[df.duplicated(keep=False)].sort_values(by='business_id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>cool</th>
      <th>date</th>
      <th>funny</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>useful</th>
      <th>user_id</th>
      <th>Review_Word_Length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1729</th>
      <td>-GY2fx-8udXPY8qn2HVBCg</td>
      <td>0</td>
      <td>2016-08-30</td>
      <td>0</td>
      <td>yQ6P1_CvM94wMLYw1T0UWA</td>
      <td>5</td>
      <td>Just opened a new account today.  So far I am ...</td>
      <td>1</td>
      <td>sZfZGrI592euyacKUcwQYg</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1729</th>
      <td>-GY2fx-8udXPY8qn2HVBCg</td>
      <td>0</td>
      <td>2016-08-30</td>
      <td>0</td>
      <td>yQ6P1_CvM94wMLYw1T0UWA</td>
      <td>5</td>
      <td>Just opened a new account today.  So far I am ...</td>
      <td>1</td>
      <td>sZfZGrI592euyacKUcwQYg</td>
      <td>55</td>
    </tr>
    <tr>
      <th>754</th>
      <td>-LRlx2j9_LB3evsRRcC9MA</td>
      <td>0</td>
      <td>2017-10-07</td>
      <td>0</td>
      <td>kUqPsZmWwLIMSstGHhWssA</td>
      <td>5</td>
      <td>The vet took the time to explain what was poss...</td>
      <td>0</td>
      <td>VgaYZ7004pTwEDSDWR6u4Q</td>
      <td>33</td>
    </tr>
    <tr>
      <th>754</th>
      <td>-LRlx2j9_LB3evsRRcC9MA</td>
      <td>0</td>
      <td>2017-10-07</td>
      <td>0</td>
      <td>kUqPsZmWwLIMSstGHhWssA</td>
      <td>5</td>
      <td>The vet took the time to explain what was poss...</td>
      <td>0</td>
      <td>VgaYZ7004pTwEDSDWR6u4Q</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2767</th>
      <td>-MKWJZnMjSit406AUKf7Pg</td>
      <td>0</td>
      <td>2015-01-03</td>
      <td>2</td>
      <td>rJhrQD3-b9GjTso0dxIkwg</td>
      <td>1</td>
      <td>Drove 37 miles on a Saturday at 12:30pm for lu...</td>
      <td>0</td>
      <td>kzP96uX8TUMmmvLtd-I3RQ</td>
      <td>18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2193</th>
      <td>zKw09ftu1730wEIZBZPoFg</td>
      <td>3</td>
      <td>2015-01-04</td>
      <td>0</td>
      <td>JV-yxKxMFp-d0rLDc_2_6w</td>
      <td>5</td>
      <td>So relaxing combined with the meditation  and ...</td>
      <td>5</td>
      <td>3mZFkwfa6XV0BBazRTva9w</td>
      <td>31</td>
    </tr>
    <tr>
      <th>496</th>
      <td>zg5rJfgT4jhzg1d6r2twnA</td>
      <td>0</td>
      <td>2014-06-21</td>
      <td>0</td>
      <td>Zbj0HgdN3AT4l-mbH-EfjA</td>
      <td>3</td>
      <td>Burger week\r\n\r\n1. Blazing Pineapple Burger...</td>
      <td>0</td>
      <td>UGW-9bbBEB3eP1o6mWD_WA</td>
      <td>62</td>
    </tr>
    <tr>
      <th>496</th>
      <td>zg5rJfgT4jhzg1d6r2twnA</td>
      <td>0</td>
      <td>2014-06-21</td>
      <td>0</td>
      <td>Zbj0HgdN3AT4l-mbH-EfjA</td>
      <td>3</td>
      <td>Burger week\r\n\r\n1. Blazing Pineapple Burger...</td>
      <td>0</td>
      <td>UGW-9bbBEB3eP1o6mWD_WA</td>
      <td>62</td>
    </tr>
    <tr>
      <th>988</th>
      <td>ziv21pDfyrgdhlrlNIgDfg</td>
      <td>0</td>
      <td>2016-08-11</td>
      <td>0</td>
      <td>fus9odxu9bjE2lSxfwNfdw</td>
      <td>5</td>
      <td>Get this!!!  Wow Karlo is amazing and best cus...</td>
      <td>2</td>
      <td>ywjqPgnMrDZKOhA33v92Cw</td>
      <td>62</td>
    </tr>
    <tr>
      <th>988</th>
      <td>ziv21pDfyrgdhlrlNIgDfg</td>
      <td>0</td>
      <td>2016-08-11</td>
      <td>0</td>
      <td>fus9odxu9bjE2lSxfwNfdw</td>
      <td>5</td>
      <td>Get this!!!  Wow Karlo is amazing and best cus...</td>
      <td>2</td>
      <td>ywjqPgnMrDZKOhA33v92Cw</td>
      <td>62</td>
    </tr>
  </tbody>
</table>
<p>666 rows × 10 columns</p>
</div>



### Remove duplicates


```python
df = df.drop_duplicates()
df.shape # Previously this was (2610, 10), now we have dropped duplicate rows
```




    (2277, 10)



### Recheck for duplicates


```python
df.duplicated().value_counts()
```




    False    2277
    dtype: int64




```python
# Duplicates should no longer exist
df[df.duplicated(keep=False)].sort_values(by='business_id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>business_id</th>
      <th>cool</th>
      <th>date</th>
      <th>funny</th>
      <th>review_id</th>
      <th>stars</th>
      <th>text</th>
      <th>useful</th>
      <th>user_id</th>
      <th>Review_Word_Length</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### Create pivot tables


```python
# This transforms the data into a person by person spreadsheet and what stars they gave various restaurants
# Most values are NaN (null or missing) because people only review a few restaurants of those that exist
usr_reviews = df.pivot(index='user_id', columns='business_id', values='stars')
usr_reviews.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>business_id</th>
      <th>-050d_XIor1NpCuWkbIVaQ</th>
      <th>-0qht1roIqleKiQkBLDkbw</th>
      <th>-3zffZUHoY8bQjGfPSoBKQ</th>
      <th>-6tvduBzjLI1ISfs3F_qTg</th>
      <th>-9nai28tnoylwViuJVrYEQ</th>
      <th>-C8sSrFqaCxp51pyo-fQLQ</th>
      <th>-Dnh48f029YNugtMKkkI-Q</th>
      <th>-FLnsWAa4AGEW4NgE8Fqew</th>
      <th>-G7MPSNBpxRJmtrJxdwt7A</th>
      <th>-GY2fx-8udXPY8qn2HVBCg</th>
      <th>...</th>
      <th>zdE82PiD6wquvjYLyhOJNA</th>
      <th>zdd3hyxB8ylYV6RcNe347Q</th>
      <th>zg5rJfgT4jhzg1d6r2twnA</th>
      <th>ziv21pDfyrgdhlrlNIgDfg</th>
      <th>zkhBU5qW_zCy0q4OEtIrsA</th>
      <th>ztP466jMUMtqLwwHqXbk9w</th>
      <th>zw9_mqWBn1QCfZg88w0Exg</th>
      <th>zwNLJ2VglfEvGu7DDZjJ4g</th>
      <th>zzYaAiC0rLNSDiFQlMKOEQ</th>
      <th>zzgSiOnuUjnBnmfR-ZG4ww</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-0biHfjE0soSptbU5G3nug</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>-2K0yp7lBT_JUOzGkpdJ_g</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>-Opvc9hAWllZSSPDUsD7NA</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>-Zdxj4wuj4D_899B7tPE3g</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>-_iULENf28RbqL2k0ja5Xw</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2192 columns</p>
</div>



## Summary

In this brief introduction, you learned the acronym ETL and got to preview a few examples of ETL processes using pandas. In the upcoming lessons you'll get a much richer understanding of these and other techniques for wrangling your data!
