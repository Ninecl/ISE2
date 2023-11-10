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

## More Experimental Results on YAGO-SE

We present more detailed experimental results on YAGO-SE here. To be specific, we report the MRR and Hits@1 results of ISE2, Grail, RED-GNN, and MBE. What's more, we also conduct ablation study and robustness analysis on YAGO-SE.

### Main Results

The MRR results of ISE2 and several competitive baselines on YAGO-SE

| Model   | s1    | s2    | s3    | s4    | s5    |
| ------- | ----- | ----- | ----- | ----- | ----- |
| MBE     | 0.540 | 0.525 | 0.483 | 0.576 | 0.583 |
| Grail   | 0.527 | 0.501 | 0.469 | 0.548 | 0.551 |
| RED-GNN | 0.544 | 0.518 | 0.472 | 0.550 | 0.583 |
| ISE2    | 0.563 | 0.528 | 0.516 | 0.592 | 0.616 |

The Hits@1 results of ISE2 and several competitive baselines on YAGO-SE

| Model   | s1    | s2    | s3    | s4    | s5    |
| ------- | ----- | ----- | ----- | ----- | ----- |
| MBE     | 0.413 | 0.392 | 0.384 | 0.433 | 0.501 |
| Grail   | 0.410 | 0.372 | 0.359 | 0.399 | 0.468 |
| RED-GNN | 0.416 | 0.386 | 0.371 | 0.415 | 0.492 |
| ISE2    | 0.441 | 0.407 | 0.401 | 0.459 | 0.513 |