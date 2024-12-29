Machine Learning and Bioinformatics Project: Inter and Intraspecies Classification of Promoter Regions of DNA

Contributers: Alex Jorjorian and Bryan Yi

Description: Promoter regions of DNA are crucial sequences that regulate the transcription and expression of genes through core sequence motifs that bind transcription factors and 
polymerases. To better understand these sequences, we developed models to classify promoter sequences in the genomes of Homo sapiens, Mus musculus, and Arabidopsis thaliana. We trained and 
tested multiple classical and neural network machine learning models on Known promoter and Non-promoter sequences for each species derived from the Eukaryotic Promoter Database. We 
hypothesized that Kmer count vectors would be sufficient input features for our machine-learning models to achieve high accuracy and AUC when generalizing from training on one species to
testing others. Our hypothesis was incorrect because all models that utilized the count vectors as features failed to produce high accuracy and AUC metrics for the Arabidopsis thaliana data 
set when trained on non-Thaliana data. However, our bespoke Convolutional Transformer neural network achieved moderately generalizable accuracy from Human onto Thaliana, utilizing raw 
sequences as features. While Kmer count vectors were insufficient to build a highly accurate and generalizable classification model for promoter sequences, sequence content alone was 
sufficient, with an appropriately complex model. 

