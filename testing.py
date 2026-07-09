"C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\.venv\Scripts\python.exe" "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\evaluate.py"
using device: cpu
using phase 2 model
f
{'='*50}
 evaluation- test set
==================================================
 loading model from : checkpoints\best_phase2.pth
[Model] Backbone: efficientnet_b0
[Model] backbone output features: 1280
[Model] GPS processor: 5--> 64 features
[Model] Classifier: 1344 -> 4 classes
[Model] Output classes: 4
[Model] Loaded trained model from checkpoints\best_phase2.pth

Loading test data..
[Dataset] Loading Data\test.csv...
[Dataset] 87,014 valid images loaded (dropped ignore)
[Dataset] Lane distibution:
    1:     85,057 (97.8)
    2:      1,848 (2.1)
  SK1:         97 (0.1)
    3:         12 (0.0)
[DataLoader] Test: 87,014 images, 1,360 batches
Running evaluation..
[Dataset] WARNING : could non load J:/Testing/250357 UHCC_25_LMD/Photos/QFM954-2025-07-04-NZST/QFM954-2025-07-04-02-34-1/250357-2025-07-04-02-23-29-887-4113.49779S-17451.86791E-107.3-1-0-QFM954---F-.jpg
[ WARN:0@50.389] global loadsave.cpp:278 cv::findDecoder imread_('J:/Testing/250357 UHCC_25_LMD/Photos/QFM954-2025-07-04-NZST/QFM954-2025-07-04-02-34-1/250357-2025-07-04-02-23-29-887-4113.49779S-17451.86791E-107.3-1-0-QFM954---F-.jpg'): can't open/read file: check file path/integrity
 processed 32,000 images..
 processed 64,000 images..

==================================================
results:
Overall accuracy: 98.37%
total test images: 87,014

 Per class accuracy:
  Lane    correct,      Total   Accuracy    Abgconf
--------------------------------------------------
    1:     84,755 /   85,057      99.64%    99.78%
    2:        813 /    1,848      43.99%    92.98%
    3:          9 /       12      75.00%    75.28%
  SK1:         18 /       97      18.56%    91.37%
 confodence analysis:
 >= 70% confidence: 99.6% auto coded,0.4% needs review
 >= 80% confidence: 99.3% auto coded,0.7% needs review
 >= 90% confidence: 98.9% auto coded,1.1% needs review
 >= 95% confidence: 98.4% auto coded,1.6% needs review

 confusion matrix (counts):
        n       1n       2n       3n     SK1
       1  84,755     295       2       5
       2   1,024     813       9       2
       3       0       3       9       0
     SK1      78       1       0      18
[evaluate] confusion matrix saved --> checkpoints\confusion_matrix.png

==================================================
SKLEARN METRICS
==================================================

Classification Report:
              precision    recall  f1-score   support

           1      0.987     0.996     0.992     85057
           2      0.731     0.440     0.549      1848
           3      0.450     0.750     0.562        12
         SK1      0.720     0.186     0.295        97

    accuracy                          0.984     87014
   macro avg      0.722     0.593     0.600     87014
weighted avg      0.981     0.984     0.982     87014

Macro metrics (unweighted average across classes):
  Macro F1:        0.600
  Macro Precision: 0.722
  Macro Recall:    0.593

Weighted metrics (weighted by class frequency):
  Weighted F1:        0.982
  Weighted Precision: 0.981
  Weighted Recall:    0.984

Comparison:
  Overall accuracy:  98.369%
  Weighted F1:       0.982  <- comparable to accuracy
  Macro F1:          0.600  <- shows class balance
  Previous team F1:  0.640            <- their best result

[evaluate] results saved --> checkpoints\evaluation_results.txt

 evaluation complete
[TensorBoard] Computing embeddings for Test_embeddings...
[ WARN:0@58.854] global loadsave.cpp:278 cv::findDecoder imread_('J:/Testing/250357 UHCC_25_LMD/Photos/QFM954-2025-07-04-NZST/QFM954-2025-07-04-02-34-1/250357-2025-07-04-02-23-29-887-4113.49779S-17451.86791E-107.3-1-0-QFM954---F-.jpg'): can't open/read file: check file path/integrity
[Dataset] WARNING : could non load J:/Testing/250357 UHCC_25_LMD/Photos/QFM954-2025-07-04-NZST/QFM954-2025-07-04-02-34-1/250357-2025-07-04-02-23-29-887-4113.49779S-17451.86791E-107.3-1-0-QFM954---F-.jpg
