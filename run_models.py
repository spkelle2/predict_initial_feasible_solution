from coinor.cuppy.milpInstance import MILPInstance
from math import isclose
import numpy as np
import os
import pandas as pd
import re
import shutil

from simple_mip_solver import BranchAndBound, PseudoCostBranchDepthFirstSearchNode as PCBDFSNode


def main():

    input_fldr = 'models'
    output_fldr = 'data'
    min_mip_gap = .2

    shutil.rmtree(output_fldr, ignore_errors=True)
    os.mkdir(output_fldr)
    pattern = re.compile('.*(\d+)\.mps')

    skipped_probs = []

    for index, file_name in enumerate(os.listdir(input_fldr)):
        print(f'solving problem {index}')
        problem = int(pattern.search(file_name).group(1))

        # solve the problem
        model = MILPInstance(file_name=f'{input_fldr}/{file_name}')
        bb = BranchAndBound(model=model, Node=PCBDFSNode, logging=False, max_runtime=60, pseudo_costs={})
        bb.solve()

        if bb.status == 'stopped on iterations or time' and bb.current_gap > min_mip_gap:
            skipped_probs.append(problem)
            continue

        data = []

        # for each solution node, collect metadata on the underlying LP
        solution_nodes = [n for n in bb.tree.get_leaves(bb.root_node.idx) if n.mip_feasible]
        for n in solution_nodes:
            row = {'objective value': n.objective_value,
                   'age': n.idx / bb.evaluated_nodes}  # DFS evaluates all nodes until gap closed
            x = n.lp.getVarByName('x')

            for j in range(n.lp.nVariables):
                # note: don't track variable type or lower bound since they're all the same
                row[f'var {j} objective coefficient'] = n.lp.objective[j]
                row[f'var {j} reduced cost'] = n.lp.dualVariableSolution['x'][j]
                row[f'var {j} value'] = n.solution[j]
                row[f'var {j} upper bound'] = n.lp.variablesUpper[j]

            for i in range(n.lp.nConstraints):
                constr_coefs = n.lp.constraints[0].varCoefs[x][i]
                row[f'constr {i} objective cosine similarity'] = np.dot(constr_coefs, n.lp.objective) / \
                    (np.linalg.norm(constr_coefs) * np.linalg.norm(n.lp.objective))
                row[f'constr {i} tight'] = int(
                    isclose(np.dot(constr_coefs, n.solution), n.lp.constraintsLower[i], abs_tol=.0001)
                )
                row[f'constr {i} dual value'] = n.lp.dualConstraintSolution[n.lp.constraints[0].name][i]

            data.append(row)

        with open(f'{output_fldr}/neural_net_inputs.csv', 'a') as f:
            pd.DataFrame(data).to_csv(f, mode='a', header=f.tell() == 0, index=False)

    print(f'\n\nskipped {len(skipped_probs)} probs')


if __name__ == '__main__':
    main()
