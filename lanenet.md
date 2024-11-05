## LaneNet
By treating lane detection as an *instance segmentation* problem.
consists of two parts(jointly trained):
- segmentation
- clustering



procedure:
- LaneNet: output a set of pixels per lane.
instance segmetation task:
  - binary lane segmentation
      - loss function: cross-entropy loss function(两类分布unbalanced，引入bounded inverser class weighting)
      - output: a one channel image
  - pixel embeddings and clustering
      - disentangle the lane pixels identified by the segmentation branch
      - output: a N-channel image with N the embedding dimension
- H-Net: the lane pixels are projected into a "bird's-eye view"(estimates the parameters of an "ideal" perspective transformation, help the lane to be fitted with a low-order polynomial)
存在疑惑？

