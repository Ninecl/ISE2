# ISE2
The source code of "Inductive Link Prediction for Sequential-emerging Knowledge Graph"

## Data
The data we used in this paper are contained in *./data/*. Especially, we provide the map file, which includes the entity ID in FB15k-237, corresponding Wiki ID, and created time in *./data/FB-SE/fb237_2w_createdtime.txt*.

## Run experiments
To train the model, you can directly using the following codes. Taking WN-SE as an example:
```
python train.py -d WN-SE -e ISE2_WN
```
To test the model, continuing the example of WN-SE:
```
python test.py -d WN-SE -e ISE2_WN
```

## Note
The efficiency of subgraph extraction is greatly improved compared with Grail and the other subgraph reasoning methods, the re-implement of subgraph extraction can be found in *./subgraph_extraction/graph_sampler.py*.