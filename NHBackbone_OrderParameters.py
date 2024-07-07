import sys
import csv
import numpy as np
import pandas as pd
from vmd import vmdnumpy, atomsel, molecule

def read_trajectory(cms_file, trajectory_file):

        molid = molecule.load("mae", cms_file)

        molecule.read(molid, "dtr", trajectory_file, first = 0, last = -1, waitfor = -1)

        totframes = molecule.numframes(molid)

        return molid, totframes

def align(molid):

###This aligns all the structures in an MD trajectory to the first frame i.e. the reference frame###

    #Total number of frames in trajectory
    totframes = molecule.numframes(molid)

    #Reference structure are the backbone heavy atoms of the first frame
    reference = atomsel('(resid 5 or resid 6 or resid 7 or resid 8 or resid 9 or resid 10 or resid 11 or resid 12 or resid 18 or resid 19 or resid 20 or resid 21 or resid 22 or resid 23 or resid 24 or resid 25 or resid 26 or resid 27 or resid 28 or resid 31 or resid 32 or resid 33 or resid 34 or resid 35 or resid 36 or resid 37 or resid 38 or resid 39 or resid 44 or resid 45 or resid 46 or resid 47 or resid 48 or resid 49 or resid 50 or resid 51 or resid 52 or resid 53 or resid 54 or resid 55 or resid 56 or resid 57 or resid 64 or resid 65 or resid 66 or resid 67 or resid 68 or resid 69 or resid 72 or resid 73 or resid 74 or resid 75 or resid 76 or resid 77 or resid 78 or resid 79 or resid 80 or resid 82 or resid 83 or resid 84 or resid 85 or resid 86 or resid 87 or resid 101 or resid 102 or resid 103 or resid 104 or resid 105 or resid 106 or resid 107 or resid 108 or resid 109 or resid 110 or resid 111 or resid 115 or resid 116 or resid 117 or resid 118 or resid 119 or resid 120 or resid 130 or resid 131 or resid 132 or resid 133 or resid 134 or resid 135 or resid 136 or resid 137 or resid 138 or resid 139 or resid 140 or resid 141) and backbone', molid = molid, frame = 0)

    #All atoms in structure
    everything = atomsel('all', molid = molid)
   # everything = atomsel("protein and backbone", molid = molid)
    #Select all backbone heavy atoms
    selection = atomsel('(resid 5 or resid 6 or resid 7 or resid 8 or resid 9 or resid 10 or resid 11 or resid 12 or resid 18 or resid 19 or resid 20 or resid 21 or resid 22 or resid 23 or resid 24 or resid 25 or resid 26 or resid 27 or resid 28 or resid 31 or resid 32 or resid 33 or resid 34 or resid 35 or resid 36 or resid 37 or resid 38 or resid 39 or resid 44 or resid 45 or resid 46 or resid 47 or resid 48 or resid 49 or resid 50 or resid 51 or resid 52 or resid 53 or resid 54 or resid 55 or resid 56 or resid 57 or resid 64 or resid 65 or resid 66 or resid 67 or resid 68 or resid 69 or resid 72 or resid 73 or resid 74 or resid 75 or resid 76 or resid 77 or resid 78 or resid 79 or resid 80 or resid 82 or resid 83 or resid 84 or resid 85 or resid 86 or resid 87 or resid 101 or resid 102 or resid 103 or resid 104 or resid 105 or resid 106 or resid 107 or resid 108 or resid 109 or resid 110 or resid 111 or resid 115 or resid 116 or resid 117 or resid 118 or resid 119 or resid 120 or resid 130 or resid 131 or resid 132 or resid 133 or resid 134 or resid 135 or resid 136 or resid 137 or resid 138 or resid 139 or resid 140 or resid 141) and backbone', molid = molid)

    #For each frame
    for i in range(totframes):

        #Select the backbone heavy atoms of that frame
        selection.frame = i

        #Select all atoms in that frame
        everything.frame = i

        #Compute the transformation matrix between the selection and reference and move everything to fit the reference
        everything.move(selection.fit(reference))

    return

def veclen(x):

        '''Returns the length of a given vector'''

        length = np.sqrt(np.sum(x**2,1))

        return length

def autocorrelation_function_score(selection1, selection2, first_frame = 0, last_frame = -1):

        '''Returns the S2 value described in the paper by Turbovic et al.'''

        #List of index numbers of your atom selections
        selection1_index = selection1.index
        selection2_index = selection2.index
        #Get the molid of your selection
        molid = selection1.molid
        #Total frames in the trajectory
        totframes = molecule.numframes(molid = molid)
        if last_frame == -1:
                last_frame = totframes
        #Create an empty list with indices referring to each frame in the trajectory i.e. index = 1 is frame = 1 etc.
        vector_permutations = [*range(int(last_frame - first_frame))]

        #Over all frames
        for j in range(int(first_frame), int(last_frame)):

                #List of all atoms with positions [x,y,z] in frame j (nx3 matrix) 
                t1 = vmdnumpy.positions(molid,j)
                #[x,y,z] of selection 1 - [x,y,z] of selection 2. Displacement vector
                x = t1[selection1_index] - t1[selection2_index]
                #Vector length of displacement vector
                xl = veclen(x)
                #Scale vectors by their length
                #For each displacement vector
                for k in range(0,len(x)):

                        #Divide vector components by vector length to get unit vector
                        x[k] = [i/xl[k] for i in x[k]]

                #All permutations of vector compnents i.e. x1x1,x1y1,x1z1....z1z1
                mu11 = [i[0]*i[0] for i in x]
                mu12 = [i[0]*i[1] for i in x]
                mu13 = [i[0]*i[2] for i in x]
                mu21 = [i[1]*i[0] for i in x]
                mu22 = [i[1]*i[1] for i in x]
                mu23 = [i[1]*i[2] for i in x]
                mu31 = [i[2]*i[0] for i in x]
                mu32 = [i[2]*i[1] for i in x]
                mu33 = [i[2]*i[2] for i in x]

                #Each frame gets a list of all vector component permutations
                vector_permutations[j-first_frame] = [mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33]

       #Mean of second level entries over all frames i.e. average mu11 over all frames.
        mean = np.mean(vector_permutations,0)

        #Square all values
        square = [i*i for i in mean]

        #Add them all together
        total_sum = sum(square)

        #Get S2
        S2 = 1.5 * total_sum - 0.5

        return S2


def order_parameter(molid, outname, totframes):

        '''A function where the user chooses which amino acid and bond vector to calculate S2 for'''

        #Amino acid in 3 letter code i.e. ARG and atoms in letter name i.e. HE, epsilon hydrogen

        amino_acid_bond_vectors = {"ARG" : ["N","H"],
                                   "MET" : ["N","(name H1 or name H)"],
                                   "LEU" : ["N","H"],
                                   "LYS" : ["N","H"],
                                 
                                   "GLN" : ["N","H"],
                                   "VAL" : ["N","H"],
                               
				   "GL4" : ["N", "H"], 
                                   "GLU" : ["N","H"],
                                 
                                   "ILE" : ["N","H"],
                                   "PHE" : ["N","H"],
                                  
                                   "TYR" : ["N","H"],
				   "AS4" : ["N","H"],
                                   "ASP" : ["N","H"],
                                   "GLY" : ["N","H"],
                                   "ASN" : ["N","H"],
                               #proline doesnt have NH bond vector
                                   "PRO" : ["CA","N"],
                                   "CYS" : ["N","H"],
				   "CYX" : ["N", "H"],
                                   "SER" : ["N","H"],
                                   "THR" : ["N","H"],
                                   "HIS" : ["N","H"],
				   "HIE" : ["N","H"],
				#Histidine with hydrogen on the dealta nitrogen
				   "HID" : ["N","H"],
				#histidine with hydrogen on the epsilon nitrogen
				   "HIE" : ["N","H"],
				#histidine with hydrogens on both nitrogens positivel charged
				   "HIP" : ["N","H"],
                                   "ALA" : ["N", "H"],
                                   "TRP" : ["N","H"],
                                   }

        blocked_order_df = pd.DataFrame()

        first = 0  
       # 50 increment = int(totframes / 40)
       #10  increment = int(totframes / 200)
        #25 increment = int(totframes / 80)
       #100  increment = int(totframes/ 20)
        #250 increment = int(totframes/8)
        #500 increment = int(totframes / 4)
        increment = int(totframes / 40)
        last = increment     
        while first < totframes:
                order_df = pd.DataFrame(columns = ["PDBID", "RESID", "RESNAME", "NH_Backbone_S2"])
                print("Block: {}".format(first))
                for aa,bv in amino_acid_bond_vectors.items():

                        selection1 = atomsel('protein and chain A and resname {0} and name {1}'.format(aa, bv[0]), molid = molid)
			
                        print("Getting S2 for {0}".format(aa))
                       # if 1 not in selection1.resid:
                        if aa == "MET":
                                selection2 = atomsel('protein and chain A and resname {0} and {1}'.format(aa, bv[1]), molid = molid)
                        else:
                                selection2 = atomsel('protein and chain A and resname {0} and name {1}'.format(aa, bv[1]), molid = molid)
                        S2 = autocorrelation_function_score(selection1, selection2, first_frame = first, last_frame = last)
         
                               

                #Write to output file
                        tmp_df = pd.DataFrame({'PDBID' : outname,
                                'RESID' : selection1.resid,
                                'RESNAME' : selection1.resname,
                                'NH_Backbone_S2': S2,
                                })
                               
                        order_df = order_df.append(tmp_df, ignore_index = True)

                if first == 0:
                        blocked_order_df = blocked_order_df.append(order_df, ignore_index = True)
                else:
                        blocked_order_df["S2_BLOCK_{}".format(first)] = order_df["NH_Backbone_S2"]
                first = int(first + increment)
                last  = int(last + increment)

                if last > totframes:
                        break
                
        blocked_order_df["MEAN_NH_Backbone_S2"] = blocked_order_df.iloc[:, 3:].mean(axis=1)
        blocked_order_df["STD_NH_Backbone_S2"] = blocked_order_df.iloc[:, 3:].std(axis=1)
        blocked_order_df.to_csv('{}-NHBackbone-all-orderparameters.csv'.format(outname))



        return 


if __name__ == "__main__":
    cms_file = sys.argv[1]
    trj_file = sys.argv[2]
    outname = sys.argv[3]
    molid, totframes = read_trajectory(cms_file, trj_file)
    align(molid)
    order_parameter(molid = molid, outname = outname, totframes = int(totframes))
    sys.exit()
