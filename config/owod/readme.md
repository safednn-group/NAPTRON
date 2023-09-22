## Towards Open World Object Detection

Reference implementation: [https://github.com/JosephKJ/OWOD](https://github.com/JosephKJ/OWOD)

As suggested by authors, this method need the following steps to be taken: 
1. Train the model on a training subset. 
2. Extract model's logits on validation subset, while the model is in training mode. (logits are to be further utilised)

The cfg files are therefore doubled.
