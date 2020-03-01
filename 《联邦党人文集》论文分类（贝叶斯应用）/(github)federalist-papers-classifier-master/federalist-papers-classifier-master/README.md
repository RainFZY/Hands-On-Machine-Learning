Federalist Papers Classifier
====================

The Federalist Papers were a series of 85 articles and essays written by Alexander Hamilton, James Madison, and John Jay, that promoted the ratification of the United States Constitution. 

However, due to the radicalness of the ideas at the time, the Federalist Papers were published anonymously, and although both Hamilton and Madison released lists detailing the author of each essay, their lists disputed the ownership of 12 of the papers.

By using K-Nearest Neighbors, a bag of words model, and semantic analysis with function words, this project has determined that Madison was the most likely author of the 12 disputed essays.

To replicate these results, clone this repo, and then run the following commands in the cloned repo.

    ~ python splitter.py
      // Splits the essays into Disputed, Madison, Hamilton, and Jay categories. 
      // Stores the splits in papers.txt.
    ~ python classifier.py
      // Classifies the 12 disputed essays. The results are printed out.
    
  