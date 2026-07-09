C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\.venv\Scripts\python.exe" "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\evaluate.py"
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
