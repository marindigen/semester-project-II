# Protein Contact Prediction
This repository details the code related to the Semester project on Contact Map prediction of simulated proteins.

## Abstract

Protein contact maps offer valuable insights into the 3D structure and function of proteins. This project explores various methods for inferring amino acid couplings. Using synthetic data generated from a minimal model, we test traditional methods and compare the results with transformer approaches trained via Masked Language Modelling (MLM). Finally, we propose an encoder-decoder transformer that shows promising results in inferring structural contacts with less computational cost.

## Structure


### Key Components:
- **data/**: Hosts all datasets utilized within the project, segregated by model complexity and real data samples.
- **model_decoder/**: Contains model-specific weights necessary for model operations and learning processes.
- **report/**: Includes detailed reports and documents summarizing the findings and progress of the project.
- **runs/**: Holds data and outputs from various experimental runs, possibly including serialized objects and logs.
- **wandb/**: Directory for integration with Weights & Biases for tracking experiments and model performance.
- **Notebooks (.ipynb)**: These interactive notebooks are used to demonstrate and test various models and algorithms in a live coding environment.

### Additional Notes:
- This structure should be maintained and updated regularly to reflect any changes in the project's organization.
- It's advisable to add more descriptive text as needed to help new users or contributors understand the function of each component in the repository.

