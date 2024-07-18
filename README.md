# Automate-Neural-relational-inference-NRI

Automate Neural Relational Inference (NRI) to accelerate the analysis of residues and ligand trajectories by learning complex residue-residue, residue-ligand, and ligand-ligand interactions.

## Requirements

- **NRI**
  - Install from their [GitHub page](https://github.com/juexinwang/NRI-MD).
  - Learn more about NRI from their article: [Nature Article](https://www.nature.com/articles/s41467-022-29331-3).
- **Python Packages**
  - pandas: `pip install pandas`
  - shutil: `pip install shutil`
  - glob: `pip install glob`

## Usage

1. Run your molecular dynamics (MD) simulations and use VMD to save all trajectories in a single PDB file.
2. Convert the simulation PDB file to an NRI-compatible format using this script from our [GitHub repository](https://github.com/ehsansyh/PDB-file-edit-for-NRI).
3. The `NRI_pdb` folder contains three sample PDB files ready for NRI. Use them to test the setup.
4. Update the paths in the `Auto_NRI` script as needed.

### Running the Auto_NRI Script

When running the `Auto_NRI` script, you will be prompted to enter:
- The number of frames obtained during MD simulations.
- The number of steps to use in NRI.
- The number of encoders and decoders to use in NRI.

### Output

- A CSV file containing the energy scores for all the MD simulations processed by NRI.
- A plot displaying the training and validation loss over epochs to monitor the model's performance.

## Cite

- **Please cite our publication: Deep Learning-Driven Discovery of FDA-Approved BCL2 Inhibitors: In Silico Analysis Using a Deep Generative Model NeuralPlexer for Drug Repurposing in Cancer Treatment (doi: https://doi.org/10.1101/2024.07.15.603544)**
