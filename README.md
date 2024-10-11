# CMSC472 Assignment 3

For Assignment 3, you will use hw3.ipynb to learn PyTorch while trying to create a bird vs. non-bird classifier. More specifically, you will:

- Write a Dataset class in PyTorch.

- Define a neural network in PyTorch.

- Implement training and testing loops in PyTorch.

- Find a way to improve the performance of your network above our baseline.

- Compete for the best performance on a held-out testset.

## Download the data file

Download the data file [here](https://umd.instructure.com/files/80453154/download?download_frd=1). Make sure that you are logged in to ELMS before downloading.

# Challenge Description

This assignment is a challenge. How exciting! As in Homework 2, this is a binary classification problem, where your network will learn to discriminate birds from other things. We have curated a special dataset, new to this assignment, consisting of birds and non-birds. As before, we provide you with train and validation images and labels. However, for the test set, we only give you test images! (No labels!) Your goal is to try to get the best classification performance in the class on these test images.


# Leaderboard 

Things just got more interesting! The top-10 testset accuracies in class will be displayed in a leaderboard which will be updated twice:
1. Intermediate round: Submit your `challenge_3_results_nickname.pth` from exercise 8 to ELMS by **Oct. 16 5.00pm**. Find the nickname rules under section 8.
2. Final round: After your final submission.

**Submissions with the top-3 accuracies in class on the *testset* during the *final round* will win lucrative ~~cash prizes~~ bonus points!**

* **Winner:** 20 bonus points 
* **Runners-up:** 10 bonus points 
* **Second Runners-up:** 5 bonus points 

**Submissions with the top-3 accuracies in class on the *testset* during the *intermediate round* will win 5 bonus points each!**


# Challenge Rules

1. You **MUST** submit a results file on ELMS. Details are provided in Section 8 of this notebook.
2. You **MAY NOT** deviate from neither of the **number of layers (2)** nor the **number of parameters (hidden layer size = 7)** in each layer from Assignment 2. 
3. You **MAY NOT** attempt to manually label the test images. We want to evaluate the performance of your neural classification network, not your classification ability.
4. You should try to achieve at least **75%** validation accuracy (Baseline).

## Submission

For Assignment 3, there are a few explaination questions asked which needs to be answered apart from the coding questions. 

Submission needs to be done on Gradescope. Upload the iPython notebook to Google Drive (do not forget to give access). Now, get the share link for the ipython notebook and embed it inside the notebook (you can do this by opening the notebook on the drive using Google Colab).  Please generate a PDF from the iPython notebook and upload it to Gradescope. Please make sure that the PDF contains the link to the Google Drive share otherwise the assignment may not be graded. **Make sure to submit the results file to ELMS as well**.

## Deadline

Assignment 3 is due on **Oct. 18 at 5:00pm**.  

None of the parts of this assignment require the use of a machine with a GPU. You may complete the assignment using your local machine or you may use Google Colab. However, we encourage you to try using Google Colab and get familiar with it as it would be helpful in upcoming assignments.

Credits: The format of this assignment is inspired by the Stanford CS231n assignments. We have borrowed some of their data loading and instructions in our assignment ipython notebook. 
