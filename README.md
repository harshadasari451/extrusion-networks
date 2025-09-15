# extrusion-networks

## Objective

The objective of this project is to study how features learned from low-dimensional data can be leveraged for high-dimensional data, minimizing the need for expensive data collection in high dimensions. For example, given 500 samples in 1D, we use pre-trained 1D weights within a 2D model via a hypernetwork approach. The repository implements two types of 2D models: 

- **Vanilla model:** A standard model trained on 100% of the available 2D data.
- **Extrusion model:** Uses a hypernetwork to transfer knowledge from 1D to 2D, requiring only a fraction of the 2D data (as little as 10–60%) to achieve performance comparable to or better than the vanilla model.

This demonstrates that by using hypernetworks and transferring features from low-dimensional to high-dimensional spaces, models can be trained efficiently with much less high-dimensional data, significantly reducing data requirements and cost.
