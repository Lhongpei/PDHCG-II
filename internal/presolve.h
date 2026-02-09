#ifndef PRESOLVE_H
#define PRESOLVE_H

#include "PSLP_API.h"
#include "PSLP_stats.h"
#include "pdhcg.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        Presolver *presolver;
        Settings *settings;
        qp_problem_t *reduced_problem;
        bool problem_solved_during_presolve;
        double presolve_time;
        char presolve_status;
    } pdhcg_presolve_info_t;

    pdhcg_presolve_info_t *pslp_presolve(const qp_problem_t *original_prob, const pdhg_parameters_t *params);

    pdhcg_result_t *create_result_from_presolve(const pdhcg_presolve_info_t *info, const qp_problem_t *original_prob);

    const char *get_presolve_status_str(enum PresolveStatus_ status);

    void pslp_postsolve(pdhcg_presolve_info_t *info,
                        pdhcg_result_t *reduced_result,
                        const qp_problem_t *original_prob);

    void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info);

#ifdef __cplusplus
}
#endif

#endif // PRESOLVE_H