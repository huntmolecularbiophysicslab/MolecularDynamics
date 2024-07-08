
EcRNH_S2_jump_model.ipynb: Jupyter Python 3 code to analyze dihedral angle distributions and calculate generalized order parameters from dihedral angles.

The code reads the following files CSV files:

1.  Torsion angles determined from MD trajectory by the nmr_orderD program (written by Kim Sharp).

2. S2 values for methyl symmetry axis determined from MD trajectory. Input files have the structure:

	Residue,Resname,Methyl,S2,S2_error

with one line per methyl group. For example (the above header is optional):

	67,LEU,CD1,0.60217625,0.10395955

3. S2 values for Ca-Cb bond vector determined from MD trajectory. The file has the structure:

	Residue,Resname,S2,S2_error

with one line per amino acid reside. For example (the above header is optional):

67,LEU,0.94334114,0.011304728

4. Experimental results for methyl 2H relaxation data analyzed using the model-free formalism. For the present work, the file is provided as EcRNH_CH3_Threefield_MCtm.csv

