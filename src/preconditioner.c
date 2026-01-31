/*
Copyright 2025 Haihao Lu

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

#include "preconditioner.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define SCALING_EPSILON 1e-12

static void scale_problem(lp_problem_t *problem, const double *con_rescale,
                          const double *var_rescale);
static void ruiz_rescaling(lp_problem_t *problem, int num_iters,
                           double *cum_con_rescale, double *cum_var_rescale);
static void pock_chambolle_rescaling(lp_problem_t *problem, double alpha,
                                     double *cum_con_rescale,
                                     double *cum_var_rescale);



lp_problem_t *deepcopy_problem(const lp_problem_t *prob)
{
    lp_problem_t *new_prob = (lp_problem_t *)safe_malloc(sizeof(lp_problem_t));

    new_prob->num_variables = prob->num_variables;
    new_prob->num_constraints = prob->num_constraints;
    new_prob->constraint_matrix_num_nonzeros =
        prob->constraint_matrix_num_nonzeros;
    new_prob->objective_matrix_num_nonzeros =
        prob->objective_matrix_num_nonzeros;
    new_prob->objective_constant = prob->objective_constant;

    size_t var_bytes = prob->num_variables * sizeof(double);
    size_t con_bytes = prob->num_constraints * sizeof(double);

    new_prob->variable_lower_bound = safe_malloc(var_bytes);
    new_prob->variable_upper_bound = safe_malloc(var_bytes);
    new_prob->objective_vector = safe_malloc(var_bytes);
    new_prob->constraint_lower_bound = safe_malloc(con_bytes);
    new_prob->constraint_upper_bound = safe_malloc(con_bytes);

    memcpy(new_prob->variable_lower_bound, prob->variable_lower_bound, var_bytes);
    memcpy(new_prob->variable_upper_bound, prob->variable_upper_bound, var_bytes);
    memcpy(new_prob->objective_vector, prob->objective_vector, var_bytes);
    memcpy(new_prob->constraint_lower_bound, prob->constraint_lower_bound,
           con_bytes);
    memcpy(new_prob->constraint_upper_bound, prob->constraint_upper_bound,
           con_bytes);
    new_prob->constraint_matrix = deepcopy_csr_component(prob->constraint_matrix, prob->num_constraints, prob->constraint_matrix_num_nonzeros);
    new_prob->objective_matrix = deepcopy_csr_component(prob->objective_matrix, prob->num_variables, prob->objective_matrix_num_nonzeros);

    if (prob->primal_start)
    {
        new_prob->primal_start = safe_malloc(var_bytes);
        memcpy(new_prob->primal_start, prob->primal_start, var_bytes);
    }
    else
    {
        new_prob->primal_start = NULL;
    }
    if (prob->dual_start)
    {
        new_prob->dual_start = safe_malloc(con_bytes);
        memcpy(new_prob->dual_start, prob->dual_start, con_bytes);
    }
    else
    {
        new_prob->dual_start = NULL;
    }

    return new_prob;
}

static void scale_problem(
    lp_problem_t *problem,
    const double *constraint_rescaling,
    const double *variable_rescaling)
{
    for (int i = 0; i < problem->num_variables; ++i)
    {
        problem->objective_vector[i] /= variable_rescaling[i];
        problem->variable_upper_bound[i] *= variable_rescaling[i];
        problem->variable_lower_bound[i] *= variable_rescaling[i];
    }
    for (int i = 0; i < problem->num_constraints; ++i)
    {
        problem->constraint_lower_bound[i] /= constraint_rescaling[i];
        problem->constraint_upper_bound[i] /= constraint_rescaling[i];
    }

    for (int row = 0; row < problem->num_constraints; ++row)
    {
        for (int nz_idx = problem->constraint_matrix->row_ptr[row];
             nz_idx < problem->constraint_matrix->row_ptr[row + 1]; ++nz_idx)
        {
            int col = problem->constraint_matrix->col_ind[nz_idx];
            problem->constraint_matrix->val[nz_idx] /= (constraint_rescaling[row] * variable_rescaling[col]);
        }
    }

    for (int q_row = 0; q_row < problem->num_variables; ++q_row)
    {
        for (int nz_idx = problem->objective_matrix->row_ptr[q_row];
             nz_idx < problem->objective_matrix->row_ptr[q_row + 1]; ++nz_idx)
        {
            int q_col = problem->objective_matrix->col_ind[nz_idx];
            problem->objective_matrix->val[nz_idx] /= (variable_rescaling[q_row] * variable_rescaling[q_col]);
        }
    }
}

static void ruiz_rescaling(
    lp_problem_t *problem,
    int num_iterations,
    double *cum_constraint_rescaling,
    double *cum_variable_rescaling)
{
    int num_cons = problem->num_constraints;
    int num_vars = problem->num_variables;
    double *con_rescale = safe_malloc(num_cons * sizeof(double));
    double *var_rescale = safe_malloc(num_vars * sizeof(double));

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        for (int i = 0; i < num_vars; ++i)
            var_rescale[i] = 0.0;
        for (int i = 0; i < num_cons; ++i)
            con_rescale[i] = 0.0;

        for (int row = 0; row < num_cons; ++row)
        {
            for (int nz_idx = problem->constraint_matrix->row_ptr[row];
                 nz_idx <  problem->constraint_matrix->row_ptr[row + 1]; ++nz_idx)
            {
                int col = problem->constraint_matrix->col_ind[nz_idx];
                if (col < 0 || col >= num_vars)
                {
                    fprintf(stderr, "Error: Invalid column index %d at nz_idx %d for row %d. Must be in [0, %d).\n",
                            col, nz_idx, row, num_vars);
                }
                double val = fabs(problem->constraint_matrix->val[nz_idx]);
                if (val > var_rescale[col])
                    var_rescale[col] = val;
                if (val > con_rescale[row])
                    con_rescale[row] = val;
            }
        }
        for (int q_row = 0; q_row < num_vars; ++q_row)
        {
            for (int nz_idx = problem->objective_matrix->row_ptr[q_row];
                 nz_idx <  problem->objective_matrix->row_ptr[q_row + 1]; ++nz_idx)
            {
                int q_col = problem->objective_matrix->col_ind[nz_idx];
                if (q_col < 0 || q_col >= num_vars)
                {
                    fprintf(stderr, "Error: Invalid column index %d at nz_idx %d for q_row %d. Must be in [0, %d).\n",
                            q_col, nz_idx, q_row, num_vars);
                }
                double val = fabs(problem->objective_matrix->val[nz_idx]);
                if (val > var_rescale[q_col])
                    var_rescale[q_col] = val;
            }
        }
        for (int i = 0; i < num_vars; ++i)
            var_rescale[i] = (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
        for (int i = 0; i < num_cons; ++i)
            con_rescale[i] = (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

        scale_problem(problem, con_rescale, var_rescale);
        for (int i = 0; i < num_vars; ++i)
            cum_variable_rescaling[i] *= var_rescale[i];
        for (int i = 0; i < num_cons; ++i)
            cum_constraint_rescaling[i] *= con_rescale[i];
    }
    free(con_rescale);
    free(var_rescale);
}

static void pock_chambolle_rescaling(
    lp_problem_t *problem,
    double alpha,
    double *cum_constraint_rescaling,
    double *cum_variable_rescaling)
{
    int num_cons = problem->num_constraints;
    int num_vars = problem->num_variables;
    double *con_rescale = safe_calloc(num_cons, sizeof(double));
    double *var_rescale = safe_calloc(num_vars, sizeof(double));

    for (int row = 0; row < num_cons; ++row)
    {
        for (int nz_idx = problem->constraint_matrix->row_ptr[row];
             nz_idx < problem->constraint_matrix->row_ptr[row + 1]; ++nz_idx)
        {
            int col = problem->constraint_matrix->col_ind[nz_idx];
            double val = fabs(problem->constraint_matrix->val[nz_idx]);
            var_rescale[col] += pow(val, 2.0 - alpha);
            con_rescale[row] += pow(val, alpha);
        }
    }

    for (int q_row = 0; q_row < num_vars; ++q_row)
    {
        for (int nz_idx = problem->objective_matrix->row_ptr[q_row];
             nz_idx < problem->objective_matrix->row_ptr[q_row + 1]; ++nz_idx)
        {
            int q_col = problem->objective_matrix->col_ind[nz_idx];
            double val = fabs(problem->objective_matrix->val[nz_idx]);
            var_rescale[q_col] += pow(val, 2.0 - alpha);
        }
    }

    for (int i = 0; i < num_vars; ++i)
        var_rescale[i] = (var_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(var_rescale[i]);
    for (int i = 0; i < num_cons; ++i)
        con_rescale[i] = (con_rescale[i] < SCALING_EPSILON) ? 1.0 : sqrt(con_rescale[i]);

    scale_problem(problem, con_rescale, var_rescale);
    for (int i = 0; i < num_vars; ++i)
        cum_variable_rescaling[i] *= var_rescale[i];
    for (int i = 0; i < num_cons; ++i)
        cum_constraint_rescaling[i] *= con_rescale[i];

    free(con_rescale);
    free(var_rescale);
}

static void bound_obj_rescaling(
    lp_problem_t *problem,
    rescale_info_t *rescale_info)
{
    double b_norm_sq = 0.0;
    for (int i = 0; i < problem->num_constraints; ++i)
    {
        if (isfinite(problem->constraint_lower_bound[i]) && (problem->constraint_lower_bound[i] != problem->constraint_upper_bound[i]))
        {
            b_norm_sq += problem->constraint_lower_bound[i] * problem->constraint_lower_bound[i];
        }
        if (isfinite(problem->constraint_upper_bound[i]))
        {
            b_norm_sq += problem->constraint_upper_bound[i] * problem->constraint_upper_bound[i];
        }
    }
    double c_norm_sq = 0.0;
    for (int i = 0; i < problem->num_variables; ++i)
    {
        c_norm_sq += problem->objective_vector[i] * problem->objective_vector[i];
    }
    rescale_info->con_bound_rescale =  sqrt(b_norm_sq) + 1.0;
    rescale_info->obj_vec_rescale =  sqrt(c_norm_sq) + 1.0;

    for (int i = 0; i < problem->num_constraints; ++i)
    {
        problem->constraint_lower_bound[i] /= rescale_info->con_bound_rescale;
        problem->constraint_upper_bound[i] /= rescale_info->con_bound_rescale;
    }
    for (int i = 0; i < problem->num_variables; ++i)
    {
        problem->variable_lower_bound[i] /= rescale_info->con_bound_rescale;
        problem->variable_upper_bound[i] /= rescale_info->con_bound_rescale;
        problem->objective_vector[i] /= rescale_info->obj_vec_rescale;
    }
    for (int nnz_idx = 0; nnz_idx < problem->objective_matrix_num_nonzeros; ++nnz_idx)
    {
        problem->objective_matrix->val[nnz_idx] *= rescale_info->con_bound_rescale / rescale_info->obj_vec_rescale;
    }
}

rescale_info_t *rescale_problem(
    const pdhg_parameters_t *params,
    const lp_problem_t *working_problem)
{
    clock_t start_rescaling = clock();
    rescale_info_t *rescale_info =
        (rescale_info_t *)safe_calloc(1, sizeof(rescale_info_t));
    rescale_info->scaled_problem = deepcopy_problem(working_problem);
    if (rescale_info->scaled_problem == NULL)
    {
        fprintf(stderr,
                "Failed to create a copy of the problem. Aborting rescale.\n");
        return NULL;
    }
    int num_cons = working_problem->num_constraints;
    int num_vars = working_problem->num_variables;

    rescale_info->con_rescale = safe_malloc(num_cons * sizeof(double));
    rescale_info->var_rescale = safe_malloc(num_vars * sizeof(double));
    
    for (int i = 0; i < num_cons; ++i)
        rescale_info->con_rescale[i] = 1.0;
    for (int i = 0; i < num_vars; ++i)
        rescale_info->var_rescale[i] = 1.0;
    
    if (params->l_inf_ruiz_iterations > 0)
    {
        ruiz_rescaling(rescale_info->scaled_problem, params->l_inf_ruiz_iterations,
                       rescale_info->con_rescale, rescale_info->var_rescale);
    }
    if (params->has_pock_chambolle_alpha)
    {
        pock_chambolle_rescaling(rescale_info->scaled_problem, params->pock_chambolle_alpha,
            rescale_info->con_rescale, rescale_info->var_rescale);
    }
    if (params->bound_objective_rescaling)
    {
        bound_obj_rescaling(rescale_info->scaled_problem, rescale_info);
    }
    else
    {
        rescale_info->con_bound_rescale = 1.0;
        rescale_info->obj_vec_rescale = 1.0;
    }
    rescale_info->rescaling_time_sec = (double)(clock() - start_rescaling) / CLOCKS_PER_SEC;
    return rescale_info;
}
