# geosolve_lanes
#lanes distribution: J:\- Macros\AI-LaneDetector\lane_analysis\coordinates_report_with_lane_combined.csv
Lane distribution:
Lane
1      2422274
2        88397
3        10933
SK1       6291
4          493
SK2        200
BK4         88
SK3         20
Name: count, dtype: int64

Number of unique segments (SourceFolder):
13

Segment distribution:
SourceFolder
250816 SouthlandDC_NetworkLMD_25             923845
250362.01 AT_Data_Collection_25_LMD          559911
250846 DunedinCC LMD Network26               332650
250724 WaitakiDC_Network_LMD25               178458
250819 SelwynDC_NetworkLMD_25_Demo           139985
250098 HCCNETWORK25_LMD                       91939
250362.05 AT_Data_Collection_25_LMD_North     74297
250728 InvercargillCC_NetworkLMD 25           67003
250821 KaikouraDC_Network25_LMD Demo          53451
250712 CluthaDC_LMD_25                        48435
Name: count, dtype: int64

Potentially stopped frames: 442335

Bearing range: 0.0 to 1000.0
Bearing distribution:
count    2.516952e+06
mean     1.821006e+02
std      1.117442e+02
min      0.000000e+00
25%      8.970000e+01
50%      1.800000e+02
75%      2.698000e+02
max      1.000000e+03
Name: Bearing, dtype: float64

=== BK4 rows ===
                                     SourceFolder  ... Bearing
709390  250362.05 AT_Data_Collection_25_LMD_North  ...   270.2
709391  250362.05 AT_Data_Collection_25_LMD_North  ...   270.2
709392  250362.05 AT_Data_Collection_25_LMD_North  ...   270.2
709393  250362.05 AT_Data_Collection_25_LMD_North  ...   270.0
709394  250362.05 AT_Data_Collection_25_LMD_North  ...   270.0

[5 rows x 3 columns]

=== Bearing anomalies ===
Rows with bearing > 360: 6141
Rows with bearing == 1000: 5901
Rows with bearing == 0: 2832

=== Stopped frames by lane ===
Lane
1      407121
2       28679
3        5335
SK1       798
4         287
SK2        64
BK4        51
SK3         0
Name: is_stopped, dtype: int64

=== SK2 and SK3 segments ===
SourceFolder
250322.01 TCC_25_LMD                   83
250846 DunedinCC LMD Network26         65
250816 SouthlandDC_NetworkLMD_25       58
250362.01 AT_Data_Collection_25_LMD    11
250357 UHCC_25_LMD                      3
Name: count, dtype: int64

=== All segments ===
SourceFolder
250816 SouthlandDC_NetworkLMD_25             923845
250362.01 AT_Data_Collection_25_LMD          559911
250846 DunedinCC LMD Network26               332650
250724 WaitakiDC_Network_LMD25               178458
250819 SelwynDC_NetworkLMD_25_Demo           139985
250098 HCCNETWORK25_LMD                       91939
250362.05 AT_Data_Collection_25_LMD_North     74297
250728 InvercargillCC_NetworkLMD 25           67003
250821 KaikouraDC_Network25_LMD Demo          53451
250712 CluthaDC_LMD_25                        48435
250357 UHCC_25_LMD                            36880
250322.01 TCC_25_LMD                          14380
250573.01 DNAPIER_LMD_26                       7462

=== Lane distribution per segment ===
SourceFolder                               Lane
250098 HCCNETWORK25_LMD                    1        84603
                                           2         6870
                                           3          299
                                           SK1        167
250322.01 TCC_25_LMD                       1        12383
                                           2         1878
                                           SK2         83
                                           SK1         21
                                           3           15
250357 UHCC_25_LMD                         1        36612
                                           2          220
                                           SK1         45
                                           SK2          3
250362.01 AT_Data_Collection_25_LMD        1       514921
                                           2        37990
                                           3         6247
                                           SK1        631
                                           4          111
                                           SK2         11
250362.05 AT_Data_Collection_25_LMD_North  1        62737
                                           2         7287
                                           3         3687
                                           4          382
                                           SK1        116
                                           BK4         88
250573.01 DNAPIER_LMD_26                   1         7232
                                           2          202
                                           SK1         28
250712 CluthaDC_LMD_25                     1        48255
                                           SK1        105
                                           2           75
250724 WaitakiDC_Network_LMD25             1       175562
                                           2         2454
                                           SK1        442
250728 InvercargillCC_NetworkLMD 25        1        61471
                                           2         5253
                                           SK1        279
250816 SouthlandDC_NetworkLMD_25           1       908837
                                           2        12210
                                           SK1       2740
                                           SK2         58
250819 SelwynDC_NetworkLMD_25_Demo         1       136234
                                           2         3578
                                           SK1        173
250821 KaikouraDC_Network25_LMD Demo       1        53257
                                           2          112
                                           SK1         82
250846 DunedinCC LMD Network26             1       320170
                                           2        10268
                                           SK1       1462
                                           3          685
                                           SK2         45
                                           SK3         20

# folder data:
"C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\.venv\Scripts\python.exe" "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\main.py" 
C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\main.py:10: DtypeWarning: Columns (0: Lane, 1: DistressDetected, 2: DetectionSource) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv(csv_file)
coordinates_report_combined_with_ignore.csv: 2,840,372 rows
coordinates_report_with_lane_combined.csv: 2,528,696 rows
coordinates_report_with_lane_combined_with_ignore.csv: 2,528,696 rows
coordinates_report_with_lane_combined_with_ignore_full.csv: 2,840,370 rows
coordinates_report_with_lane_combined_with_ignore_full1.csv: 2,840,370 rows
merged_laneFixes.csv: 28,575 rows
per_project_stats.csv: 26 rows
per_session_stats.csv: 154 rows
SK1.csv: 6,291 rows

Total rows across all jobs: 13,613,550

All Lane codes found:
Lane
1      10394546
2        372337
3         45087
SK1       44709
4          2006
SK2        1678
BK4         357
SK3         343
-1            6
5             5

All segments found:
SourceFolder
250816 SouthlandDC_NetworkLMD_25             1850430
250362.01 AT_Data_Collection_25_LMD          1120453
250846 DunedinCC LMD Network26                666762
250724 WaitakiDC_Network_LMD25                357358
250819 SelwynDC_NetworkLMD_25_Demo            280143
250098 HCCNETWORK25_LMD                       184045
250362.05 AT_Data_Collection_25_LMD_North     148710
250728 InvercargillCC_NetworkLMD 25           134285
250821 KaikouraDC_Network25_LMD Demo          106984
250712 CluthaDC_LMD_25                         96975
250357 UHCC_25_LMD                             73805
250322.01 TCC_25_LMD                           28781
250573.01 DNAPIER_LMD_26                       14952


"C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\.venv\Scripts\python.exe" "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\main.py" 
Traceback (most recent call last):
  File "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\main.py", line 25, in <module>
    combined = pd.concat(all_dfs, ignore_index = True)
  File "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\.venv\Lib\site-packages\pandas\core\reshape\concat.py", line 407, in concat
    objs, keys, ndims = _clean_keys_and_objs(objs, keys)
                        ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\.venv\Lib\site-packages\pandas\core\reshape\concat.py", line 808, in _clean_keys_and_objs
    raise ValueError("No objects to concatenate")
ValueError: No objects to concatenate

Process finished with exit code 1




# dataset output:
"C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\.venv\Scripts\python.exe" "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\dataset.py" 
C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\dataset.py:19: UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise
  A.GaussNoise(var_limit= (10.0,50.0), p=0.3),
testing dataset.py...
========================================
[Dataset] Loading Data\train.csv...
[Dataset] 2,378,739 valid images loaded (dropped ignore)
[Dataset] Lane distibution:
    1:  2,278,852 (95.8)
    2:     84,589 (3.6)
    3:     10,721 (0.5)
  SK1:      4,577 (0.2)
[DataLoader] Train : 2,378,739 images, 594,684 batches per epoch

Batch shapes:
    images:      torch.Size([4, 3, 224, 224])
    gps features: torch.Size([4, 5])
    labels:        torch.Size([4])
 f
Label values: [0, 0, 0, 0]

Image tensor states:
 min:        -2.118
 max:        2.640
 mean: -0.265

GPS features (forst image):
lanes_forward : 1.000
is_oneway : 0.000
road_type : 0.000
bearing_sin : 0.000
bearing_cos : 1.000

dataset.py isn working correctly :)

Process finished with exit code 0


# model.py testing 
"C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\.venv\Scripts\python.exe" "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\model.py" 
Testing model.py..
========================================
[Model] Backbone: efficientnet_b0
[Model] backbone output features: 1280
[Model] GPS processor: 5--> 64 features
[Model] Classifier: 1344 -> 4 classes
[Model] Output classes: 4
[Model] total parameters: 4,764,672
[Model] trainable parameters: 4,764,672

Input shapes:
images: torch.Size([4, 3, 224, 224])
GPS  torch.Size([4, 5])
 output shape: torch.Size([4, 4])

 sources for first image:
 1: 0.0311
 2: -0.0008
 3: -0.0837
 SK1: 0.0701

 predicted class: 3

 --- freeze/ unfreeze test--
[Model] Backbone FROZEN - only classifier trains
[Model] total parameters: 4,764,672
[Model] trainable parameters: 757,124
[Model] backbone UNFROZEN- full fine tuining active
[Model] total parameters: 4,764,672
[Model] trainable parameters: 4,764,672

 predicte test --
 Image 1: {'lane': 'REVIEW', 'confidence': 0.267, 'needs_review': True}
 Image 2: {'lane': 'REVIEW', 'confidence': 0.268, 'needs_review': True}
 Image 3: {'lane': 'REVIEW', 'confidence': 0.266, 'needs_review': True}
 Image 4: {'lane': 'REVIEW', 'confidence': 0.266, 'needs_review': True}

model.py works correctly :)

Process finished with exit code 0


# train.py testing: 
"C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\.venv\Scripts\python.exe" "C:\Users\gd\New folder\project\geosolve_lanes\geosolve_lanes\train.py" 
GEOSOLVE lane detection- training
==================================================
QUICK TEST - 2 btaches only
========================================
device: cpu
[Dataset] Loading Data\train.csv...
[Dataset] 2,378,739 valid images loaded (dropped ignore)
[Dataset] Lane distibution:
    1:  2,278,852 (95.8)
    2:     84,589 (3.6)
    3:     10,721 (0.5)
  SK1:      4,577 (0.2)
[DataLoader] Train : 2,378,739 images, 594,684 batches per epoch
[Dataset] Loading Data\val.csv...
[Dataset] 53,248 valid images loaded (dropped ignore)
[Dataset] Lane distibution:
    1:     53,058 (99.6)
    2:        112 (0.2)
  SK1:         78 (0.1)
[DataLoader] val: 53,248 images, 13,312 batches
[Model] Backbone: efficientnet_b0
[Model] backbone output features: 1280
[Model] GPS processor: 5--> 64 features
[Model] Classifier: 1344 -> 4 classes
[Model] Output classes: 4
 batch 1 | loss: 1.3221
 batch 2 | loss: 1.0387
 vla batch 1 | loss: 1.2227 | correct: 4/4
 vla batch 2 | loss: 1.2227 | correct: 4/4

 quick test passed - training loop work correctly! 

Process finished with exit code 0