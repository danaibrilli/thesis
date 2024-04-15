### Scene-Graph guided Video Question Answering

This thesis presents a novel approach by integrating Scene Graphs with a Hierarchical Conditional approach to efficiently answer questions about videos. Scene graphs provide a structured representation of the visual elements within a video and their interrelations, offering a rich semantic foundation for understanding complex video data. By transforming the video analysis from pixel to graph space we enable more efficient and semantically rich video processing.

We propose an architecture that leverages scene graphs, utilizes Graph Neural Networks (GNNs) for processing scene graphs, alongside a hierarchical model that operates at different levels of video granularity, from individual clips to the entire video, to enable a comprehensive understanding of video content. The integration of GNNs allows for the extraction of meaningful graph embeddings that capture the relationships and attributes of the visual elements, leading to a deeper understanding of the video content. The hierarchical model, operating at different levels, ensures that both the details and the broader context are considered, leading to a more holistic understanding of the video content.

So, we introduce a methodology that (1) Begins with the extraction of scene graphs from selected video frames, (2) Generates graph embeddings using GNNs and (3) Incorporates the graph embeddings into a hierarchical model. 

We evaluate our method on the Action Genome Question Answering Dataset, a real- world dataset consisting of videos depicting humans in everyday activities. Our results demonstrate that our approach is among state-of-the-art methods, and even outperforms them in several question categories. Our approach is a step towards more efficient and context-aware Video Question Answering systems, enabling more accurate and meaningful responses to natural language queries about videos.

### Features

- **CRN (Conditional Relational Network)**: Provides methods like relation set creation and forward propagation of neural network units.
- **GraphQADataset**: Manages loading and handling of the dataset specifically formatted for graph-based video question answering.
- **HCRN (Hierarchical Conditional Relation Network)**: Incorporates various types of input and output units to handle diverse VQA tasks, utilizing features from the `CRN.py` module.
- **Input Units**: Handles different types of input data including linguistic and visual cues, and integrates graph data for enriched context understanding.
- **Residual EdgeGATConv**: An implementation of Graph Attention Networks with residual connections, enhancing feature learning from graph data.
- **Scene Model (SCENE)**: A model to process Scene Graph information using Enhanced Graph Attention Network techniques.
- **Train Functions**: Includes utility functions for model training such as data loading, batch processing, and metrics calculations.

### Installation
The project utilizes Python with various libraries such as `torch`, `dgl` (Deep Graph Library), `numpy` and `joblib`. You will need to set up a Python environment and install these packages to run the models.

### Usage
The main execution happens through `train.py`, which includes comprehensive functions for training the models, evaluating them, and visualizing results. Use commands like the following to run training sessions:

```bash
python train.py --config config.yml --model CRN
```

### Dependencies
- PyTorch (torch)
- DGL (Deep Graph Library)
- NumPy
- Joblib

### Configuration
Your configuration settings for training and model parameters are likely managed via a YAML file (`config_test.yml`). These settings include model choices, dataset paths, learning rates, etc.

### Documentation and Examples
While specific in-code documentation was not analyzed, each class and function should be well-documented within the code files for ease of understanding and usage. Example usage of models can be extrapolated from the `train.py` script.