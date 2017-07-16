# sentiment-analysis-machine
This algorithm tries to handle the problem statement 'Learning Word Vectors for Sentiment Analysis' described in wvSent_acl2011.pdf
using HD Computing. 
## Dataset
The dataset used for this projects is same as the data set used in the project described in the paper above. It uses set of IMDB reviews
divided as positive and negative. This is used to train the model and a seperate, similarily prepared, dataset is used to test the model. 
## trained_data_10000_5.mat
This containes the trained data that synthesis 10000-D vetors using 5-grams words.
## trained_data_10000_6.mat
This containes the trained data that synthesis 10000-D vetors using 6-grams words.
## accuracy 
For 5 grams words the accuracy was 68% 
For 6 grams words the accuracy was 69%
If prediction of each review was made randomly then the accuracy would have been 50%. 
This shows that the prepared model using HD computing might not have an amaxing accuracy but it is able to capture the sentiment of text 
to some extent using the HD computing model. 

