Machine Learning and Bioinformatics Project: Inter and Intraspecies Classification of Promoter Regions of DNA

Contributers: 
- Alex Jorjorian
- Bryan Yi

Description: 
Promoter regions of DNA are crucial sequences that regulate the transcription and expression of genes through core sequence motifs that bind transcription factors and 
polymerases. To better understand these sequences, we developed models to classify promoter sequences in the genomes of Homo sapiens, Mus musculus, and Arabidopsis thaliana. We trained and 
tested multiple classical and neural network machine learning models on Known promoter and Non-promoter sequences for each species derived from the Eukaryotic Promoter Database. We 
hypothesized that Kmer count vectors would be sufficient input features for our machine-learning models to achieve high accuracy and AUC when generalizing from training on one species to
testing others. Our hypothesis was incorrect because all models that utilized the count vectors as features failed to produce high accuracy and AUC metrics for the Arabidopsis thaliana
data set when trained on non-Thaliana data. However, our bespoke Convolutional Transformer neural network achieved moderately generalizable accuracy from Human onto Thaliana, utilizing raw 
sequences as features. While Kmer count vectors were insufficient to build a highly accurate and generalizable classification model for promoter sequences, sequence content alone was 
sufficient, with an appropriately complex model. 

Usage:
1. Download the two Jupyter Notebooks in the 'Code' folder named "Final_Project_Code.ipynb" and "ML_Final_Cross_Species_Analysis.ipynb"
2. Download the necessary files from the 'Data' folder and ensure that the file paths in "ML_Final_Cross_Species_Analysis.ipynb" match the location of the files in your computer
   - araTha1.txt
   - mm10.txt
   - hg38.txt
   - arabidopsis_epdnew_KZQd3.bed
   - mouse_epdnew_HlytC.bed
   - human_epdnew_Lyu0l.bed
3. Run the "ML_Final_Cross_Species_Analysis.ipynb" code and ensure the following txt files are produced:
   - ara_tha_final.csv
   - mouse_mus_final.csv
   - human_final.csv
4. Run the "Final_Project_Code.ipynb" using the txt files produced from step #3. 
