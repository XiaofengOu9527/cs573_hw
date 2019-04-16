# cs573_hw2

Name: Xiaofeng Ou <br />

Instructions to run the code in terminal:<br />

Preprocessing:<br /> python preprocess.py dating-full.csv dating.csv <br />

Visualization:<br /> 1) python 2_1.py dating.csv <br />
               2) python 2_2.py dating.csv
               <br /> <br />
Discretize: <br />python discretize.py dating.csv 5 dating-binned.csv <br />

Split:<br /> python split.py dating-binned.csv trainingSet.csv testSet.csv <br />

Model training & evaluation:<br />
               1) python 5_1.py trainingSet.csv testSet.csv 1 <br />
               2) python 5_2.py <br />
               3) python 5_3.py <br />
