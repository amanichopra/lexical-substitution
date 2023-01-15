# Lexical Substitution

In this project, the goal is to find lexical substitutes for individual target words in context. I build 4 different predictors using WordNet, WSD (word sense disambiguation), pre-trained Word2Vec, and BERT. 

First, I built a predictor using WordNet. I extract candidate words using WordNet by finding the synsets the target word appears in and then extract the words contained in each synset. Then, I use compute frequency counts for all the senses if the word and target appear in multiple synsets. Finally, I select the word with the highest frequency.

Next, I built a predictor using WSD. I look at all the synsets the word appears in and then compute the overlap between the context of the target and the definition/examples the synset contains. Then, I select the synset with the largest overlap and use the most frequent lexeme from the synset as the lexical substitute.

Third, I built a predictor using Word2Vec. I use the WordNet candidates obtained in the WordNet predictor to find the candidate most similar to the target word using the pretrained embeddings. 

Lastly, I built a predictor using BERT. Since BERT is trained on MLM objective, it's perfect for this task. I replace the target word with the ```MASK``` token and then use BERT to output predictions for the ```MASK```. Then, I use the prediction with the highest probability that appears in the WordNet candidates obtained in the WordNet predictor.

All 4 predictors can be run using ```python3 lexsub_main.py```.
