# Vulnerability Analysis of the European Natural Gas Grid using Graph Theory Methodologies

This repository contains the source files necessary for producing the results from "Vulnerability Analysis of the European Natural Gas Grid using Graph Theory Methodologies".

## Directories

- `data`: Various data compiled for use in EDA
- `IGGIELGN`: Data set from the SciGRID_gas project
- `graph_objects`: NetworkX graph objects produced from the processed SciGRID_gas project data
- `results_tables`: Pickle files with the complete results from all analyses

The scripts are stored in the root folder.

## Usage

Running the `run_results.ipynb` file will automatically produce all the results and store them in the `results_tables` folder. The results are already stored there for reference.

All data processing is documented in `data_preprocessing.ipynb`. `topology_analysis.ipynb` contains the topological analysis.

## Contributing

Feel free to contribute to this project by submitting an issue.

## License

The code in this project is available under the [MIT license](https://choosealicense.com/licenses/mit/).
