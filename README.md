# Audio Classification KNN
This program uses time domain features to classify WAV files using a K-Nearest Neighbours (KNN) algorithm. 

It extracts mean amplitude, standard deviation, Root Mean Square (RMS) energy and Zero-Crossing Rate (ZCR)
from the WAV files, and uses them to predict which class a new audio file belongs to.

It normalizes the features to have zero mean and unit variance. It also splits data into training (80%) and 
test (20%) sets using stratified sampling.

Each folder name becomes a class label, and each WAV file in that folder is a training sample for that class.

REQUIRES: Python 3, numpy, librosa, Collections (standard library)

OUTPUTS:
- Dataset loading info
- Training/test split info
- Accuracy results for different k values (k = 1,3,5,7)
- Accuracy for each class
- Results of test file classification
