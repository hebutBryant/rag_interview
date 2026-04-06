## Freebase Dataset Preparation



1. Download data

   ```bash
   wget https://download.microsoft.com/download/A/E/4/AE428B7A-9EF9-446C-85CF-D8ED0C9B1F26/FastRDFStore-data.zip --no-check-certificate
   ```

2. Process data

   ```bash
   python -m dataset.freebase
   ```

3. Dataset Statistics

   | KG       | #Entities   | #Relations | #Triplets   |
   | -------- | ----------- | ---------- | ----------- |
   | Freebase | 100,176,641 | 658,641    | 298,458,255 |




## CWQ Dataset Preparation

1. Download data: https://drive.google.com/drive/folders/18H2JXDFPWfe4WeSXVpf1lUxdpraB2OvW
2. Process data
   ```bash
   python -m dataset.cwq
   ```
3. Dataset Statistics: CWQ has 34,689 questions




## MetaQA Dataset Preparation

1. Download data: https://github.com/yuyuz/MetaQA
2. Process data

   ```bash
   python -m dataset.metaqa
   ```

3. Dataset Statistics

   | Dataset Split | 1-hop   | 2-hop   | 3-hop   |
   | ------------- | -----   | -----   | -----   |
   | Train         | 96,106  | 118,980 | 114,196 |
   | Dev           | 9,992   | 14,872  | 14,274  |
   | Test          | 9,947   | 14,872  | 14,274  |
   | All           | 11,6045 | 148,724 | 142,744 |






## WebQSP Dataset Preparation

1. Download data: https://www.microsoft.com/en-us/download/details.aspx?id=52763
2. Process data

   ```bash
   python -m dataset.webqsp
   ```

3. Dataset Statistics

   | Dataset Split | #Questions  |
   | ------------- | ----------- |
   |      Train    |    3,098    |
   |      Test     |    1,639    |
   |      All      |    4,737    |

## WebQuestions Dataset Preparation



1. Download data: https://nlp.stanford.edu/software/sempre/
2. Process data

   ```bash
   python -m dataset.webquestions
   ```

3. Dataset Statistics

   | Dataset Split | #Questions  |
   | ------------- | ----------- |
   |      Train    |    3,778    |
   |      Test     |    2,032    |
   |      All      |    5,810    |



## Grail Dataset Preparation



1. Download data: https://dl.orangedox.com/WyaCpL/
2. Process data

   ```bash
   python -m dataset.grailqa
   ```

3. Dataset Statistics

   | Dataset Split   | #Questions |
   |-----------------|-----------:|
   | Train           |     44,337 |
   | Dev             |      6,763 |
   | Test (Public)   |     13,231 |
   | All             |     64,331 |
