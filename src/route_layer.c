#include "route_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include <stdio.h>

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes, int groups, int group_id)
{
    fprintf(stderr,"route ");
    route_layer l = { (LAYER_TYPE)0 };
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    l.groups = groups;
    l.group_id = group_id;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    outputs = outputs / groups;
    l.outputs = outputs;
    l.inputs = outputs;
    //fprintf(stderr, " inputs = %d \t outputs = %d, groups = %d, group_id = %d \n", l.inputs, l.outputs, l.groups, l.group_id);
    l.delta = (float*)xcalloc(outputs * batch, sizeof(float));
    l.output = (float*)xcalloc(outputs * batch, sizeof(float));
    l.fix_output = (float*)xcalloc(l.outputs * batch, sizeof(float));

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;
    #ifdef GPU
    l.forward_gpu = forward_route_layer_gpu;
    l.backward_gpu = backward_route_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    #endif
    return l;
}

void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("Error: Different size of input layers: %d x %d, %d x %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
            exit(EXIT_FAILURE);
        }
    }
    l->out_c = l->out_c / l->groups;
    l->outputs = l->outputs / l->groups;
    l->inputs = l->outputs;
    l->delta = (float*)xrealloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = (float*)xrealloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif

}

void forward_route_layer(const route_layer l, network_state state)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = state.net.layers[index].output;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            //copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
            copy_cpu(part_input_size, input + j*input_size + part_input_size*l.group_id, 1, l.output + offset + j*l.outputs, 1);
        }
        //offset += input_size;
        offset += part_input_size;
    }

    if (l.bitwidth)
    {   
        if(l.quantized_switch & 2)
        {
            float max_data = (powf(2, l.bitwidth - 1) - 1);
            float min_data = (-powf(2, l.bitwidth - 1) + 1);
            float scale = powf(2, *l.max_out);
            for(int i = 0 ; i < l.outputs*l.batch; i++){
                l.output[i] *=scale;
                l.output[i] = fmax(fmin(l.output[i],max_data), min_data);
                l.output[i] = round (l.output[i]) ;
                l.output[i] = l.output[i] / scale;    
            }
        }
    }
}

void backward_route_layer(const route_layer l, network_state state)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = state.net.layers[index].delta;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            //axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
            axpy_cpu(part_input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size + part_input_size*l.group_id, 1);
        }
        //offset += input_size;
        offset += part_input_size;
    }
}

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network_state state)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = state.net.layers[index].output_gpu;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            //copy_ongpu(input_size, input + j*input_size, 1, l.output_gpu + offset + j*l.outputs, 1);
            //simple_copy_ongpu(input_size, input + j*input_size, l.output_gpu + offset + j*l.outputs);
            simple_copy_ongpu(part_input_size, input + j*input_size + part_input_size*l.group_id, l.output_gpu + offset + j*l.outputs);
        }
        //offset += input_size;
        offset += part_input_size;
    }


    if (l.bitwidth)
    {   
        if(l.quantized_switch & 1)
        {
            float delta_max = 0;
            float second_max = 0;

            cudaMemcpy(l.fix_output, l.output_gpu, l.outputs*l.batch*sizeof(float), cudaMemcpyDeviceToHost);
            for(int i = 0; i < l.outputs*l.batch; i++)
            {
                if(delta_max < abs(l.fix_output[i]))
                {
                    second_max = delta_max;
                    delta_max = abs(l.fix_output[i]);
                }
            }

            if(second_max > *l.max_value_out)
                 *l.max_value_out = second_max;
        }
        if(l.quantized_switch & 2)
        {
            Trim2FixedPoint_gpu(l.outputs*l.batch, 0, l.output_gpu, l.output_gpu, 1, l.bitwidth, 0, *l.max_out);
        }
    }
}

void backward_route_layer_gpu(const route_layer l, network_state state)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = state.net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            //axpy_ongpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
            axpy_ongpu(part_input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size + part_input_size*l.group_id, 1);
        }
        //offset += input_size;
        offset += part_input_size;
    }
}
#endif
