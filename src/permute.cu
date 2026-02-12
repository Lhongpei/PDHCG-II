/*
Copyright 2025 Haihao Lu
Copyright 2026 Hongpei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "permute.h"
#include "utils.h"
#include <math.h>
#include <random>

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

int cmp_tuples(const void *a, const void *b) {
    return ((permute_tuple_t *)a)->new_col - ((permute_tuple_t *)b)->new_col;
}

void col_permute_in_place(
    int m,                             
    int *Ap, int *Aj, double *Ax,       
    const int *old_col_to_new          
) {
    int max_row_nnz = 0;
    for(int i=0; i<m; i++) {
        int len = Ap[i+1] - Ap[i];
        if(len > max_row_nnz) max_row_nnz = len;
    }
    
    permute_tuple_t *buffer = (permute_tuple_t *)malloc(max_row_nnz * sizeof(permute_tuple_t));

    for (int r = 0; r < m; r++) {
        int start = Ap[r];
        int end = Ap[r+1];
        int len = end - start;

        if (len == 0) continue;

        for (int k = 0; k < len; k++) {
            int current_idx = start + k;
            int old_col = Aj[current_idx];
            
            buffer[k].new_col = old_col_to_new[old_col];
            buffer[k].val = Ax[current_idx];
        }

        if (len > 1) {
            qsort(buffer, len, sizeof(permute_tuple_t), cmp_tuples);
        }

        for (int k = 0; k < len; k++) {
            int current_idx = start + k;
            Aj[current_idx] = buffer[k].new_col;
            Ax[current_idx] = buffer[k].val;
        }
    }

    free(buffer);
}

void permute_rows_structural(qp_problem_t *qp, const int *row_perm) {
    int m = qp->num_constraints;
    int nnz = qp->constraint_matrix_num_nonzeros;
    
    int *new_Ap = (int *)malloc((m + 1) * sizeof(int));
    int *new_Aj = (int *)malloc(nnz * sizeof(int));
    double *new_Ax = (double *)malloc(nnz * sizeof(double));

    new_Ap[0] = 0;
    int current_nz = 0;

    for (int i = 0; i < m; i++) {
        int old_row_idx = row_perm[i]; 
        
        int start = qp->constraint_matrix->row_ptr[old_row_idx];
        int len = qp->constraint_matrix->row_ptr[old_row_idx + 1] - start;
        
        if (len > 0) {
            memcpy(&new_Aj[current_nz], &qp->constraint_matrix->col_ind[start], len * sizeof(int));
            memcpy(&new_Ax[current_nz], &qp->constraint_matrix->val[start], len * sizeof(double));
            current_nz += len;
        }
        
        new_Ap[i + 1] = current_nz;
    }

    free(qp->constraint_matrix->row_ptr);
    free(qp->constraint_matrix->col_ind);
    free(qp->constraint_matrix->val);

    qp->constraint_matrix->row_ptr = new_Ap;
    qp->constraint_matrix->col_ind = new_Aj;
    qp->constraint_matrix->val = new_Ax;
}

void permute_double_array(double *arr, int n, const int *perm) {
    if (!arr) return;
    double *tmp = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) tmp[i] = arr[perm[i]];
    memcpy(arr, tmp, n * sizeof(double));
    free(tmp);
}

void compute_inv_perm(int n, const int *perm, int *inv_perm) {
    for (int i = 0; i < n; i++) inv_perm[perm[i]] = i;
}

void permute_problem(qp_problem_t *qp, int *row_perm, int *col_perm) {
    int m = qp->num_constraints;
    int n = qp->num_variables;

    permute_double_array(qp->objective_vector, n, col_perm);
    permute_double_array(qp->variable_lower_bound, n, col_perm);
    permute_double_array(qp->variable_upper_bound, n, col_perm);
    permute_double_array(qp->primal_start, n, col_perm);

    permute_double_array(qp->constraint_lower_bound, m, row_perm);
    permute_double_array(qp->constraint_upper_bound, m, row_perm);
    permute_double_array(qp->dual_start, m, row_perm);

    permute_rows_structural(qp, row_perm);

    int *inv_col_perm = (int *)malloc(n * sizeof(int));
    compute_inv_perm(n, col_perm, inv_col_perm);

    col_permute_in_place(
        m, 
        qp->constraint_matrix->row_ptr, 
        qp->constraint_matrix->col_ind, 
        qp->constraint_matrix->val, 
        inv_col_perm
    );

    free(inv_col_perm);
}

qp_problem_t *permute_problem_return_new(const qp_problem_t *qp, int *row_perm, int *col_perm) {
    if (!qp) return NULL;

    qp_problem_t *new_qp = (qp_problem_t *)malloc(sizeof(qp_problem_t));
    if (!new_qp) return NULL;

    new_qp = deepcopy_problem(qp);

    #undef DEEP_COPY_ARRAY
    permute_problem(new_qp, row_perm, col_perm);

    return new_qp;
}

void generate_random_permutation(int n, int *perm) {
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
}

void randomly_permute_problem(qp_problem_t *qp, int **out_row_perm, int **out_col_perm) {
    int m = qp->num_constraints;
    int n = qp->num_variables;

    int *row_perm = (int *)malloc(m * sizeof(int));
    int *col_perm = (int *)malloc(n * sizeof(int));

    generate_random_permutation(m, row_perm);
    generate_random_permutation(n, col_perm);

    permute_problem(qp, row_perm, col_perm);

    *out_row_perm = row_perm;
    *out_col_perm = col_perm;
}


void generate_block_permutation(int n, int block_size, int *perm) {
    if (block_size <= 0) block_size = 1; 
    int num_blocks = (n + block_size - 1) / block_size;

    int *block_indices = (int *)malloc(num_blocks * sizeof(int));
    if (!block_indices) return; 

    for (int i = 0; i < num_blocks; i++) {
        block_indices[i] = i;
    }

    for (int i = num_blocks - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = block_indices[i];
        block_indices[i] = block_indices[j];
        block_indices[j] = temp;
    }

    int current_pos = 0;
    
    for (int i = 0; i < num_blocks; i++) {
        int b_idx = block_indices[i];
        
        int start_val = b_idx * block_size;
        int end_val = MIN((b_idx + 1) * block_size, n);

        for (int val = start_val; val < end_val; val++) {
            perm[current_pos++] = val;
        }
    }
    free(block_indices);
}

void randomly_block_permute_problem(qp_problem_t *qp, 
                                    int row_block_size, 
                                    int col_block_size, 
                                    int **out_row_perm, 
                                    int **out_col_perm) {
    int m = qp->num_constraints;
    int n = qp->num_variables;

    int *row_perm = (int *)malloc(m * sizeof(int));
    int *col_perm = (int *)malloc(n * sizeof(int));

    generate_block_permutation(m, row_block_size, row_perm);
    generate_block_permutation(n, col_block_size, col_perm);

    permute_problem(qp, row_perm, col_perm);

    *out_row_perm = row_perm;
    *out_col_perm = col_perm;
}