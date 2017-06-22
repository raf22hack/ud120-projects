--- email_preprocess.py	(original)
+++ email_preprocess.py	(refactored)
@@ -1,7 +1,7 @@
 #!/usr/bin/python
 
 import pickle
-import cPickle
+import pickle
 import numpy
 
 from sklearn import cross_validation
@@ -33,7 +33,7 @@
     authors_file_handler.close()
 
     words_file_handler = open(words_file, "r")
-    word_data = cPickle.load(words_file_handler)
+    word_data = pickle.load(words_file_handler)
     words_file_handler.close()
 
     ### test_size is the percentage of events assigned to the test set
@@ -58,7 +58,7 @@
     features_test_transformed  = selector.transform(features_test_transformed).toarray()
 
     ### info on the data
-    print "no. of Chris training emails:", sum(labels_train)
-    print "no. of Sara training emails:", len(labels_train)-sum(labels_train)
+    print("no. of Chris training emails:", sum(labels_train))
+    print("no. of Sara training emails:", len(labels_train)-sum(labels_train))
     
     return features_train_transformed, features_test_transformed, labels_train, labels_test
