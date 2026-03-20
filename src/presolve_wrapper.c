/*
 * PDHCG-II PSQP Presolve Wrapper
 */

#ifdef PSQP_AVAILABLE

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "PSQP_API.h"
#include "PSQP_sol.h"
#include "PSQP_status.h"
#include "presolve_wrapper.h"
#include "utils.h"

#ifndef PSQP_VERSION
#define PSQP_VERSION "unknown"
#endif

static void *safe_calloc_wrapper(size_t num, size_t size)
{
    void *ptr = calloc(num, size);
    if (!ptr && num > 0 && size > 0)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

const char *pdhcg_get_presolve_status_str(int status)
{
    switch (status)
    {
        case UNCHANGED:
            return "UNCHANGED";
        case REDUCED:
            return "REDUCED";
        case INFEASIBLE:
            return "INFEASIBLE";
        case UNBNDORINFEAS:
            return "INFEASIBLE_OR_UNBOUNDED";
        default:
            return "UNKNOWN_STATUS";
    }
}

static qp_problem_t *convert_psqp_to_pdhcg(PresolvedProblem *reduced_prob, double original_obj_constant)
{
    if (!reduced_prob)
        return NULL;

    qp_problem_t *pdhcg_prob = (qp_problem_t *)safe_calloc_wrapper(1, sizeof(qp_problem_t));

    pdhcg_prob->num_variables = (int)reduced_prob->n;
    pdhcg_prob->num_constraints = (int)reduced_prob->m;
    pdhcg_prob->constraint_matrix_num_nonzeros = (int)reduced_prob->nnz;
    pdhcg_prob->objective_constant = original_obj_constant + reduced_prob->obj_offset;

    /* Copy vector data from PSQP to avoid use-after-free when presolver is freed */
    if (reduced_prob->c)
    {
        pdhcg_prob->objective_vector = (double *)safe_malloc(reduced_prob->n * sizeof(double));
        memcpy(pdhcg_prob->objective_vector, reduced_prob->c, reduced_prob->n * sizeof(double));
    }
    if (reduced_prob->lhs)
    {
        pdhcg_prob->constraint_lower_bound = (double *)safe_malloc(reduced_prob->m * sizeof(double));
        memcpy(pdhcg_prob->constraint_lower_bound, reduced_prob->lhs, reduced_prob->m * sizeof(double));
    }
    if (reduced_prob->rhs)
    {
        pdhcg_prob->constraint_upper_bound = (double *)safe_malloc(reduced_prob->m * sizeof(double));
        memcpy(pdhcg_prob->constraint_upper_bound, reduced_prob->rhs, reduced_prob->m * sizeof(double));
    }
    if (reduced_prob->lbs)
    {
        pdhcg_prob->variable_lower_bound = (double *)safe_malloc(reduced_prob->n * sizeof(double));
        memcpy(pdhcg_prob->variable_lower_bound, reduced_prob->lbs, reduced_prob->n * sizeof(double));
    }
    if (reduced_prob->ubs)
    {
        pdhcg_prob->variable_upper_bound = (double *)safe_malloc(reduced_prob->n * sizeof(double));
        memcpy(pdhcg_prob->variable_upper_bound, reduced_prob->ubs, reduced_prob->n * sizeof(double));
    }

    if (reduced_prob->nnz > 0 && reduced_prob->Ap)
    {
        pdhcg_prob->constraint_matrix = (CsrComponent *)safe_calloc_wrapper(1, sizeof(CsrComponent));
        /* Copy PSQP's matrix data to avoid use-after-free when presolver is freed */
        pdhcg_prob->constraint_matrix->row_ptr = (int *)safe_malloc((reduced_prob->m + 1) * sizeof(int));
        memcpy(pdhcg_prob->constraint_matrix->row_ptr, reduced_prob->Ap, (reduced_prob->m + 1) * sizeof(int));
        pdhcg_prob->constraint_matrix->col_ind = (int *)safe_malloc(reduced_prob->nnz * sizeof(int));
        memcpy(pdhcg_prob->constraint_matrix->col_ind, reduced_prob->Ai, reduced_prob->nnz * sizeof(int));
        pdhcg_prob->constraint_matrix->val = (double *)safe_malloc(reduced_prob->nnz * sizeof(double));
        memcpy(pdhcg_prob->constraint_matrix->val, reduced_prob->Ax, reduced_prob->nnz * sizeof(double));
    }

    if (reduced_prob->has_quadratic && reduced_prob->Pnnz > 0 && reduced_prob->Pp)
    {
        pdhcg_prob->objective_sparse_matrix = (CsrComponent *)safe_calloc_wrapper(1, sizeof(CsrComponent));
        /* Copy PSQP's matrix data to avoid use-after-free when presolver is freed */
        pdhcg_prob->objective_sparse_matrix->row_ptr = (int *)safe_malloc((reduced_prob->n + 1) * sizeof(int));
        memcpy(pdhcg_prob->objective_sparse_matrix->row_ptr, reduced_prob->Pp, (reduced_prob->n + 1) * sizeof(int));
        pdhcg_prob->objective_sparse_matrix->col_ind = (int *)safe_malloc(reduced_prob->Pnnz * sizeof(int));
        memcpy(pdhcg_prob->objective_sparse_matrix->col_ind, reduced_prob->Pi, reduced_prob->Pnnz * sizeof(int));
        pdhcg_prob->objective_sparse_matrix->val = (double *)safe_malloc(reduced_prob->Pnnz * sizeof(double));
        memcpy(pdhcg_prob->objective_sparse_matrix->val, reduced_prob->Px, reduced_prob->Pnnz * sizeof(double));
        pdhcg_prob->objective_sparse_matrix_num_nonzeros = (int)reduced_prob->Pnnz;
    }

    return pdhcg_prob;
}

static void free_converted_problem(qp_problem_t *prob)
{
    if (!prob)
        return;
    free(prob->objective_vector);
    free(prob->constraint_lower_bound);
    free(prob->constraint_upper_bound);
    free(prob->variable_lower_bound);
    free(prob->variable_upper_bound);
    if (prob->constraint_matrix)
    {
        free(prob->constraint_matrix->row_ptr);
        free(prob->constraint_matrix->col_ind);
        free(prob->constraint_matrix->val);
        free(prob->constraint_matrix);
    }
    if (prob->objective_sparse_matrix)
    {
        free(prob->objective_sparse_matrix->row_ptr);
        free(prob->objective_sparse_matrix->col_ind);
        free(prob->objective_sparse_matrix->val);
        free(prob->objective_sparse_matrix);
    }
    free(prob->objective_lowrank_matrix);
    free(prob);
}

pdhcg_presolve_info_t *pdhcg_presolve(const qp_problem_t *original_prob, const pdhg_parameters_t *params)
{
    if (!original_prob)
        return NULL;

    if (original_prob->num_constraints == 0)
    {
        if (params->verbose > 1)
        {
            printf("Note: Problem has no constraints, skipping presolve.\n");
        }
        return NULL;
    }

    clock_t start_time = clock();

    pdhcg_presolve_info_t *info = (pdhcg_presolve_info_t *)calloc(1, sizeof(pdhcg_presolve_info_t));
    if (!info)
        return NULL;

    Settings *settings = default_settings();
    if (!settings)
    {
        free(info);
        return NULL;
    }
    set_settings_false(settings);
    info->settings = settings;

    bool has_qr = (original_prob->objective_sparse_matrix != NULL && original_prob->objective_lowrank_matrix != NULL &&
                   original_prob->num_rank_lowrank_obj > 0 && original_prob->objective_lowrank_matrix_num_nonzeros > 0);
    bool has_p = (original_prob->objective_sparse_matrix != NULL);

    Presolver *presolver = NULL;
    size_t m = (size_t)original_prob->num_constraints;
    size_t n = (size_t)original_prob->num_variables;
    size_t nnz = (size_t)original_prob->constraint_matrix_num_nonzeros;

    if (has_qr)
    {
        size_t Pnnz = (size_t)original_prob->objective_sparse_matrix_num_nonzeros;
        size_t Rnnz = (size_t)original_prob->objective_lowrank_matrix_num_nonzeros;
        size_t k = (size_t)original_prob->num_rank_lowrank_obj;
        presolver =
            new_qp_presolver_qr(original_prob->constraint_matrix ? original_prob->constraint_matrix->val : NULL,
                                original_prob->constraint_matrix ? original_prob->constraint_matrix->col_ind : NULL,
                                original_prob->constraint_matrix ? original_prob->constraint_matrix->row_ptr : NULL,
                                m,
                                n,
                                nnz,
                                original_prob->constraint_lower_bound,
                                original_prob->constraint_upper_bound,
                                original_prob->variable_lower_bound,
                                original_prob->variable_upper_bound,
                                original_prob->objective_vector,
                                original_prob->objective_sparse_matrix->val,
                                original_prob->objective_sparse_matrix->col_ind,
                                original_prob->objective_sparse_matrix->row_ptr,
                                Pnnz,
                                original_prob->objective_lowrank_matrix->val,
                                original_prob->objective_lowrank_matrix->col_ind,
                                original_prob->objective_lowrank_matrix->row_ptr,
                                Rnnz,
                                k,
                                settings);
    }
    else if (has_p)
    {
        size_t Pnnz = (size_t)original_prob->objective_sparse_matrix_num_nonzeros;
        presolver =
            new_qp_presolver(original_prob->constraint_matrix ? original_prob->constraint_matrix->val : NULL,
                             original_prob->constraint_matrix ? original_prob->constraint_matrix->col_ind : NULL,
                             original_prob->constraint_matrix ? original_prob->constraint_matrix->row_ptr : NULL,
                             m,
                             n,
                             nnz,
                             original_prob->constraint_lower_bound,
                             original_prob->constraint_upper_bound,
                             original_prob->variable_lower_bound,
                             original_prob->variable_upper_bound,
                             original_prob->objective_vector,
                             original_prob->objective_sparse_matrix->val,
                             original_prob->objective_sparse_matrix->col_ind,
                             original_prob->objective_sparse_matrix->row_ptr,
                             Pnnz,
                             settings);
    }
    else
    {
        presolver = new_presolver(original_prob->constraint_matrix ? original_prob->constraint_matrix->val : NULL,
                                  original_prob->constraint_matrix ? original_prob->constraint_matrix->col_ind : NULL,
                                  original_prob->constraint_matrix ? original_prob->constraint_matrix->row_ptr : NULL,
                                  m,
                                  n,
                                  nnz,
                                  original_prob->constraint_lower_bound,
                                  original_prob->constraint_upper_bound,
                                  original_prob->variable_lower_bound,
                                  original_prob->variable_upper_bound,
                                  original_prob->objective_vector,
                                  settings);
    }

    if (!presolver)
    {
        free_settings(settings);
        free(info);
        return NULL;
    }
    info->presolver = presolver;

    PresolveStatus status = run_presolver(presolver);
    info->presolve_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    info->presolve_status = (int)status;

    if (params->verbose > 1)
    {
        printf("\nRunning presolver (PSQP %s)...\n", PSQP_VERSION);
        printf("  %-15s : %s\n", "status", pdhcg_get_presolve_status_str(status));
        printf("  %-15s : %.3g sec\n", "presolve time", info->presolve_time);
        if (presolver->reduced_prob)
        {
            printf("  %-15s : %zu rows, %zu columns, %zu nonzeros\n",
                   "reduced problem",
                   presolver->reduced_prob->m,
                   presolver->reduced_prob->n,
                   presolver->reduced_prob->nnz);
        }
    }

    // Check if problem was solved during presolve (infeasible, unbounded, or all vars fixed)
    // Note: reduced_prob can be NULL if presolve detected infeasibility early
    bool presolver_solved = (status & INFEASIBLE) || (status & UNBNDORINFEAS) ||
        (presolver->reduced_prob && presolver->reduced_prob->n == 0);

    if (presolver_solved)
    {
        info->problem_solved_during_presolve = true;
        info->reduced_problem = NULL;
    }
    else
    {
        info->problem_solved_during_presolve = false;
        info->reduced_problem = convert_psqp_to_pdhcg(presolver->reduced_prob, original_prob->objective_constant);
    }

    return info;
}

pdhcg_result_t *pdhcg_create_result_from_presolve(const pdhcg_presolve_info_t *info, const qp_problem_t *original_prob)
{
    if (!info || !info->presolver)
        return NULL;

    Presolver *presolver = (Presolver *)info->presolver;

    pdhcg_result_t *result = (pdhcg_result_t *)safe_calloc_wrapper(1, sizeof(pdhcg_result_t));

    result->num_variables = original_prob->num_variables;
    result->num_constraints = original_prob->num_constraints;
    result->num_nonzeros = original_prob->constraint_matrix_num_nonzeros;

    // Safely access reduced_prob - it may be NULL if presolve detected infeasibility
    if (presolver->reduced_prob)
    {
        result->num_reduced_variables = (int)presolver->reduced_prob->n;
        result->num_reduced_constraints = (int)presolver->reduced_prob->m;
        result->num_reduced_nonzeros = (int)presolver->reduced_prob->nnz;
    }
    else
    {
        result->num_reduced_variables = 0;
        result->num_reduced_constraints = 0;
        result->num_reduced_nonzeros = 0;
    }
    result->presolve_status = info->presolve_status;
    result->presolve_time = info->presolve_time;

    if (info->presolve_status == INFEASIBLE)
    {
        result->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
    }
    else if (info->presolve_status == UNBNDORINFEAS)
    {
        result->termination_reason = TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED;
    }
    else if (presolver->reduced_prob && presolver->reduced_prob->n == 0)
    {
        result->termination_reason = TERMINATION_REASON_OPTIMAL;
    }
    else
    {
        result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    }

    if (result->num_variables > 0)
    {
        result->primal_solution = (double *)safe_calloc_wrapper(result->num_variables, sizeof(double));
        result->reduced_cost = (double *)safe_calloc_wrapper(result->num_variables, sizeof(double));
    }
    if (result->num_constraints > 0)
    {
        result->dual_solution = (double *)safe_calloc_wrapper(result->num_constraints, sizeof(double));
    }

    // If presolve solved the problem directly, copy solution from presolver->sol
    // This happens when reduced problem has 0 variables (n == 0)
    // Note: reduced_prob may be NULL if presolve detected infeasibility
    if (presolver->reduced_prob && presolver->reduced_prob->n == 0 && presolver->sol)
    {
        if (result->primal_solution && presolver->sol->x)
        {
            memcpy(result->primal_solution, presolver->sol->x, result->num_variables * sizeof(double));
        }
        if (result->dual_solution && presolver->sol->y)
        {
            memcpy(result->dual_solution, presolver->sol->y, result->num_constraints * sizeof(double));
        }
        if (result->reduced_cost && presolver->sol->z)
        {
            memcpy(result->reduced_cost, presolver->sol->z, result->num_variables * sizeof(double));
        }

        // When all variables are fixed, objective value is:
        // original_constant + obj_offset (which includes all fixed var contributions)
        double primal_obj = original_prob->objective_constant + presolver->reduced_prob->obj_offset;
        result->primal_objective_value = primal_obj;
        result->dual_objective_value = primal_obj;
    }

    return result;
}

void pdhcg_postsolve(const pdhcg_presolve_info_t *info, pdhcg_result_t *result, const qp_problem_t *original_prob)
{
    if (!info || !info->presolver || !result)
        return;

    Presolver *presolver = (Presolver *)info->presolver;

    double *primal_sol = (double *)safe_calloc_wrapper(original_prob->num_variables, sizeof(double));
    double *dual_sol = (double *)safe_calloc_wrapper(original_prob->num_constraints, sizeof(double));
    double *reduced_cost = (double *)safe_calloc_wrapper(original_prob->num_variables, sizeof(double));

    if (result->primal_solution)
    {
        memcpy(primal_sol, result->primal_solution, result->num_variables * sizeof(double));
    }
    if (result->dual_solution)
    {
        memcpy(dual_sol, result->dual_solution, result->num_constraints * sizeof(double));
    }
    if (result->reduced_cost)
    {
        memcpy(reduced_cost, result->reduced_cost, result->num_variables * sizeof(double));
    }

    if (result->primal_solution)
        free(result->primal_solution);
    if (result->dual_solution)
        free(result->dual_solution);
    if (result->reduced_cost)
        free(result->reduced_cost);

    result->primal_solution = primal_sol;
    result->dual_solution = dual_sol;
    result->reduced_cost = reduced_cost;

    // When n == 0, solution was already recovered in pdhcg_presolve
    // Skip all postsolve processing to avoid overwriting correct values
    // Note: reduced_prob may be NULL if presolve detected infeasibility
    if (presolver->reduced_prob && presolver->reduced_prob->n == 0)
    {
        // Solution is already in result->primal_solution from pdhcg_create_result_from_presolve
        // Just update the metadata
        result->num_reduced_variables = (int)presolver->reduced_prob->n;
        result->num_reduced_constraints = (int)presolver->reduced_prob->m;
        result->num_reduced_nonzeros = (int)presolver->reduced_prob->nnz;
        result->presolve_status = info->presolve_status;
        result->presolve_time = info->presolve_time;
        return;
    }

    postsolve(presolver, result->primal_solution, result->dual_solution, result->reduced_cost);

    if (presolver->sol)
    {
        memcpy(result->primal_solution, presolver->sol->x, original_prob->num_variables * sizeof(double));
        memcpy(result->dual_solution, presolver->sol->y, original_prob->num_constraints * sizeof(double));
        memcpy(result->reduced_cost, presolver->sol->z, original_prob->num_variables * sizeof(double));
    }

    // Safely access reduced_prob - it may be NULL if presolve detected infeasibility
    if (presolver->reduced_prob)
    {
        result->num_reduced_variables = (int)presolver->reduced_prob->n;
        result->num_reduced_constraints = (int)presolver->reduced_prob->m;
        result->num_reduced_nonzeros = (int)presolver->reduced_prob->nnz;
    }
    else
    {
        result->num_reduced_variables = 0;
        result->num_reduced_constraints = 0;
        result->num_reduced_nonzeros = 0;
    }
    result->presolve_status = info->presolve_status;
    result->presolve_time = info->presolve_time;

    for (int i = 0; i < original_prob->num_variables; i++)
    {
        if (!isfinite(original_prob->variable_lower_bound[i]))
        {
            result->reduced_cost[i] = fmin(result->reduced_cost[i], 0.0);
        }
        if (!isfinite(original_prob->variable_upper_bound[i]))
        {
            result->reduced_cost[i] = fmax(result->reduced_cost[i], 0.0);
        }
    }
}

void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info)
{
    if (!info)
        return;

    if (info->reduced_problem)
    {
        free_converted_problem(info->reduced_problem);
    }

    if (info->presolver)
    {
        free_presolver((Presolver *)info->presolver);
    }

    if (info->settings)
    {
        free_settings((Settings *)info->settings);
    }

    free(info);
}

const char *pdhcg_presolve_version(void)
{
    return "PSQP " PSQP_VERSION;
}

int pdhcg_presolve_available(void)
{
    return 1;
}

#else /* PSQP_AVAILABLE */

#include "presolve_wrapper.h"
#include <stdio.h>
#include <stdlib.h>

const char *pdhcg_get_presolve_status_str(int status)
{
    (void)status;
    return "PSQP_NOT_AVAILABLE";
}

pdhcg_presolve_info_t *pdhcg_presolve(const qp_problem_t *original_prob, const pdhg_parameters_t *params)
{
    (void)original_prob;
    (void)params;
    fprintf(stderr, "Warning: PSQP not available, presolving disabled.\n");
    return NULL;
}

pdhcg_result_t *pdhcg_create_result_from_presolve(const pdhcg_presolve_info_t *info, const qp_problem_t *original_prob)
{
    (void)info;
    (void)original_prob;
    return NULL;
}

void pdhcg_postsolve(const pdhcg_presolve_info_t *info, pdhcg_result_t *result, const qp_problem_t *original_prob)
{
    (void)info;
    (void)result;
    (void)original_prob;
}

void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info)
{
    (void)info;
}

const char *pdhcg_presolve_version(void)
{
    return "PSQP not available";
}

int pdhcg_presolve_available(void)
{
    return 0;
}

#endif /* PSQP_AVAILABLE */
