#ifndef PDHCG_DISTRIBUTED_INTERFACE_H
#define PDHCG_DISTRIBUTED_INTERFACE_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    struct grid_context_s;
    typedef struct grid_context_s grid_context_t;

    typedef enum
    {
        PDHCG_SCOPE_GLOBAL,
        PDHCG_SCOPE_ROW,
        PDHCG_SCOPE_COL
    } pdhcg_comm_scope_t;

    typedef enum
    {
        PDHCG_OP_SUM,
        PDHCG_OP_MAX
    } pdhcg_reduce_op_t;

    void pdhcg_all_reduce_array(
        grid_context_t *ctx, double *buf, int count, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, void *stream);

    void pdhcg_all_reduce_scalar(
        grid_context_t *ctx, double *value, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, bool on_device);

    int pdhcg_get_grid_p_col(struct grid_context_s *ctx);

    int pdhcg_get_grid_row_coord(struct grid_context_s *ctx);

#ifdef __cplusplus
}
#endif

#endif // PDHCG_DISTRIBUTED_INTERFACE_H
