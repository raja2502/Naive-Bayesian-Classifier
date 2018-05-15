import arff
from sklearn.naive_bayes import GaussianNB
import codecs
import weka.core.jvm as jvm 
jvm.start()

from weka.classifiers import Evaluation, Classifier
from weka.core.classes import Random
from weka.core.converters import Loader
import weka.core.serialization as serialization
import sys

def case1() :
    loader = Loader(classname="weka.core.converters.ArffLoader")
    file = input("Enter the name of the file without the extension:")
    data = loader.load_file(file+".arff", incremental=True)
    data.class_is_last()
	# print(data)
	# print(str(data1))
    cls = Classifier(classname="weka.classifiers.bayes.NaiveBayesUpdateable")
    cls.build_classifier(data)
    for inst in loader:
        cls.update_classifier(inst)
    serialization.write(file+".bin", cls)
    print("Model created with name:",file,".bin")

def case2() :
    loader1 = Loader(classname="weka.core.converters.ArffLoader")
    file = input("Enter the name of the  model file:")
    cls2 = Classifier(jobject=serialization.read(file))
    test_file = input("Enter the name of the test file:")
    data1=loader1.load_file(test_file)
    data1.class_is_last()
    evaluation = Evaluation(data1)
    evl = evaluation.test_model(cls2, data1)
    print(evaluation.matrix("=== (confusion matrix) ==="))

def case3() :
    # print("Nothing yet macha!")
    file_ = codecs.open('weather.nominal.arff', 'rb', 'utf-8')
    data = arff.load(file_,encode_nominal=True)
    gnb = GaussianNB()
    no_of_rows=0
    data2 = []
    target = []
    data1 = data['data']
    for row in data1:
        target.append(row.pop())
        data2.append(row)
    gnb.fit(data2,target)
    while(1):
        ch = int(input("1.Enter values interactively 2.Quit :"))
        if ch == 1 :
           outlook=int(input("Please enter a value for Outlook.\n0.Sunny 1.Overcast 2.Rainy:"))
           temperature=int(input("Please enter a value for Temperature.\n0.Hot 1.Mild 2.Cool:"))
           humidity=int(input("Please enter a value for Outlook.\n0.High 1.Normal:"))		   
           windy=int(input("Please enter a value for Windy.\n0.True 1.False :"))		   
           # print(gnb)
           predict=int(gnb.predict([[outlook,temperature,humidity,windy,]]))
           if predict == 1 :
               print("The Prediction is:No")
           else :
               print("The Prediction is:Yes") 		   
        else :
           break


while(1):
    print("========================================================================================================")
    print("1. Learn a Naïve Bayesian classifier from data. \n2. Load and test accuracy of a naïve Bayesian classifier. \n3. Apply a naïve Bayesian classifier to new cases. \n4. Quit.")
    print("========================================================================================================")
    choice = int(input("Enter your choice:"))
    while choice not in [1,2,3,4]:
        choice = int(input("Invalid choice.\n 1. Learn a Naïve Bayesian classifier from data. \n2. Load and test accuracy of a naïve Bayesian classifier. \n3. Apply a naïve Bayesian classifier to new cases. \n4. Quit."))
    if choice == 1 :
        case1()
    elif choice == 2 :
        case2()
    elif choice == 3 :
        case3()
    else :
        print("The program will now terminate!")
        jvm.stop()		
        sys.exit()
    


#print(cls)
jvm.stop()