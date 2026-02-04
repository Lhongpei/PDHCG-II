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

#pragma once

#include "internal_types.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C"
{
#endif

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess)                                    \
        {                                                          \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cudaGetErrorName(err));              \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                           \
    do                                                               \
    {                                                                \
        cublasStatus_t status = call;                                \
        if (status != CUBLAS_STATUS_SUCCESS)                         \
        {                                                            \
            fprintf(stderr, "cuBLAS Error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cublasGetStatusName(status));          \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

#define CUSPARSE_CHECK(call)                                           \
    do                                                                 \
    {                                                                  \
        cusparseStatus_t status = call;                                \
        if (status != CUSPARSE_STATUS_SUCCESS)                         \
        {                                                              \
            fprintf(stderr, "cuSPARSE Error at %s:%d: %s\n", __FILE__, \
                    __LINE__, cusparseGetErrorName(status));           \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define THREADS_PER_BLOCK 256
#define ALLOC_AND_COPY(dest, src, bytes)  \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));

#define ALLOC_AND_COPY_CSR(dest_csr, src_csr, n_rows, nnz)             \
    do {                                                               \
        ALLOC_AND_COPY((dest_csr)->row_ptr,                            \
                       (src_csr)->row_ptr,                             \
                       ((n_rows) + 1) * sizeof(int));                  \
                                                                       \
        ALLOC_AND_COPY((dest_csr)->col_ind,                            \
                       (src_csr)->col_ind,                             \
                       (nnz) * sizeof(int));                           \
                                                                       \
        ALLOC_AND_COPY((dest_csr)->val,                                \
                       (src_csr)->val,                                 \
                       (nnz) * sizeof(double));                        \
    } while (0)

#define ALLOC_ZERO(dest, bytes)           \
    CUDA_CHECK(cudaMalloc(&dest, bytes)); \
    CUDA_CHECK(cudaMemset(dest, 0, bytes));
    
    extern const double HOST_ONE;
    extern const double HOST_ZERO;

    void *safe_malloc(size_t size);

    void *safe_calloc(size_t num, size_t size);

    void *safe_realloc(void *ptr, size_t new_size);

    
    lp_problem_t *deepcopy_problem(const lp_problem_t *prob);

    double estimate_maximum_singular_value(
        cusparseHandle_t sparse_handle,
        cublasHandle_t blas_handle,
        const cu_sparse_matrix_csr_t *A,
        const cu_sparse_matrix_csr_t *AT,
        int max_iterations,
        double tolerance);

    double estimate_maximum_eigenvalue(cusparseHandle_t sparse_handle,
        cublasHandle_t blas_handle,
        const cu_sparse_matrix_csr_t *A,
        int max_iterations, double tolerance);

    double estimate_minimum_eigenvalue(cusparseHandle_t sparse_handle,
        cublasHandle_t blas_handle,
        const cu_sparse_matrix_csr_t *A,
        double lambda_max, 
        int max_iterations, double tolerance);

    void compute_interaction_and_movement(
        pdhg_solver_state_t *solver_state,
        double *interaction,
        double *movement);

    bool should_do_adaptive_restart(
        pdhg_solver_state_t *solver_state,
        const restart_parameters_t *restart_params,
        int termination_evaluation_frequency);

    void check_termination_criteria(
        pdhg_solver_state_t *solver_state,
        const termination_criteria_t *criteria);

    void print_initial_info(const pdhg_parameters_t *params, const lp_problem_t *problem);

    void pdhg_final_log(const cupdlpx_result_t *result, const pdhg_parameters_t *params);

    void display_iteration_stats(const pdhg_solver_state_t *solver_state, bool verbose);

    const char *termination_reason_to_string(termination_reason_t reason);
    const char *problem_type_to_string(problem_type_t type);
    const char *quad_obj_type_to_string(quad_obj_type_t type);

    int get_print_frequency(int iter);

    void compute_residual(pdhg_solver_state_t *state, norm_type_t optimality_norm);

    void compute_infeasibility_information(pdhg_solver_state_t *state);

    void fill_or_copy(double **dest, int n, const double *src, double fill_value);

    int dense_to_csr(const matrix_desc_t *desc,
                     int **row_ptr, int **col_ind, double **vals, int *nnz_out);

    int csc_to_csr(const matrix_desc_t *desc,
                   int **row_ptr, int **col_ind, double **vals, int *nnz_out);

    int coo_to_csr(const matrix_desc_t *desc,
                   int **row_ptr, int **col_ind, double **vals, int *nnz_out);

    void check_feas_polishing_termination_criteria(
        pdhg_solver_state_t *solver_state,
        const termination_criteria_t *criteria,
        bool is_primal_polish);

    void print_initial_feas_polish_info(bool is_primal_polish, const pdhg_parameters_t *params);

    void display_feas_polish_iteration_stats(const pdhg_solver_state_t *state, bool verbose, bool is_primal_polish);

    void pdhg_feas_polish_final_log(const pdhg_solver_state_t *primal_state, const pdhg_solver_state_t *dual_state, bool verbose);

    void compute_primal_feas_polish_residual(pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state, norm_type_t optimality_norm);

    void compute_dual_feas_polish_residual(pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state, norm_type_t optimality_norm);

    void set_default_parameters(pdhg_parameters_t *params);

    __global__ void compute_residual_kernel(
        double *primal_residual, const double *primal_product,
        const double *constraint_lower_bound, const double *constraint_upper_bound,
        const double *dual_solution, double *dual_residual,
        const double *dual_product, const double *dual_slack,
        const double *objective_vector, const double *constraint_rescaling,
        const double *variable_rescaling, double *dual_obj_contribution,
        const double *const_lb_finite, const double *const_ub_finite,
        int num_constraints, int num_variables);

    double get_vector_sum(cublasHandle_t handle, int n, double *ones_d,
                             const double *x_d);
    double get_vector_inf_norm(cublasHandle_t handle, int n,
                                  const double *x_d);
    
    __global__ void primal_infeasibility_project_kernel(
        double *primal_ray_estimate, const double *variable_lower_bound,
        const double *variable_upper_bound, int num_variables);
    __global__ void dual_infeasibility_project_kernel(
        double *dual_ray_estimate, const double *constraint_lower_bound,
        const double *constraint_upper_bound, int num_constraints);
    __global__ void compute_dual_infeasibility_kernel(const double *dual_product,
                                    const double *var_lb, const double *var_ub,
                                    int num_variables, double *dual_infeasibility,
                                  const double *variable_rescaling);
    __global__ void dual_solution_dual_objective_contribution_kernel(
        const double *constraint_lower_bound_finite_val,
        const double *constraint_upper_bound_finite_val,
        const double *dual_solution, int num_constraints,
        double *dual_objective_dual_solution_contribution_array);
    
    __global__ void dual_objective_dual_slack_contribution_array_kernel(
        const double *dual_slack,
        double *dual_objective_dual_slack_contribution_array,
        const double *variable_lower_bound_finite_val,
        const double *variable_upper_bound_finite_val, int num_variables);
        
    __global__ void compute_primal_infeasibility_kernel(
        const double *primal_product, const double *const_lb,
        const double *const_ub, int num_constraints, double *primal_infeasibility,
        const double *constraint_rescaling);
    CsrComponent *deepcopy_csr_component(const CsrComponent *src, size_t num_rows, size_t nnz);
    quad_obj_type_t detect_q_type(const CsrComponent *csr, int num_rows);
    void ensure_objective_matrix_initialized(lp_problem_t *prob);
#ifdef __cplusplus
}

#endif
