// __author__ = Tarek Samaali

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
//#include <windows.h>
#include <ctype.h>
#include <stdint.h>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#define UPPER_BOUND .5

#define GEN_RANDOM_SEED srand(time(NULL))
#define DEBUG printf("debug !")
#define ERROR_DIM_BLOCK_DIMENSIONS printf("\nBlocks must have same dimensions .. \n");
#define NULL_BLOCK printf("\nSorry .. the block is null ..\n")
#define ERROR_DIMENSION_CONV printf("Dimension conflict: Please review the dimensions of the convolved arrays! \n")
#define ERROR_DEPTH printf("Cannot perform convolution: Please make sure the kernel and the block have the same depth. \n")
#define ERROR_NULL printf("Null input. \n")
#define ERROR_CREATING_BLOCKS printf("Cannot create blocks : please make sure the length is > 1.\n")
#define ERROR_EVEN_DIMENSIONS printf(" How about using a kernel with an odd size :))) ! .. ")
#define ERROR_DIMENSION_GRID_MULT printf("Please review the dimensions of the grid you want to perform multiplication on.\n")
#define ERROR_DIM_FLATTEN printf("Input must be flattened.\n")
#define ERROR_MODEL printf("\nPlease initialize the model before starting to use it.\n")

#define current_Layer(x) printf("\nCurrent Layer: %s\n",x)
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))
#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define write(x) printf("\n%s\n",x)

//#define add__(a,b) ({retun a+b;})
//#define substract__(a,b) ({retun a-b;})


typedef enum{

    CONV,
    POOL,
    FLATTEN,
    FULLY_CONNECTED_AFTER_FLATTEN,
    FULLY_CONNECTED,
    ACTIVATION__

}TYPE_LAYER;


char* getType(TYPE_LAYER name){

        if(name==CONV)
            return "CONVOLUTION";
        if(name==POOL)
            return "POOLING";
        if(name==FLATTEN)
            return "FLATTEN";
        if(name==FULLY_CONNECTED_AFTER_FLATTEN)
            return "FULLY_CONNECTED_AFTER_FLATTEN";
        if(name==FULLY_CONNECTED)
            return "FULLY_CONNECTED";
        if(name==ACTIVATION__)
            return "ACTIVATION";

}


//Define parameters and then construct union over the parameters
struct params_CONV{

    double (*activation__)(double);
    uint32_t stride;
    uint32_t padding;
    uint32_t nbr_filters;
    uint32_t kernel_size;

    TYPE_LAYER name;

};


struct params_POOL{

    uint32_t stride;
    uint32_t padding;
    char* pooling_choice;
    uint32_t kernel_size;

    TYPE_LAYER name;

};


struct params_FLATTEN{

    TYPE_LAYER name;

};


struct params_FCAF{
    //Params for Fully connected after Flatten
    double (*activation__)(double);
    uint32_t output_size;

    TYPE_LAYER name;

};


struct params_FC{

    double (*activation__)(double);
    uint32_t output_size;

    TYPE_LAYER name;

};


typedef struct params_CONV paramsCONV;
typedef struct params_POOL paramsPOOL;
typedef struct params_FLATTEN paramsFLATTEN;
typedef struct params_FCAF paramsFCAF;
typedef struct params_FC paramsFC;


//2D output
//After single convolution
typedef struct {
   uint32_t width;
   uint32_t height;
    double** grid;

} Grid;

//3D Block for a single
typedef struct{

   uint32_t depth;
   uint32_t width;
   uint32_t height;
    double*** matrix;

}Block;

//4D output
//After convolution with N filters
typedef struct{

   uint32_t length;
    Block** blocks;

}Blocks;



typedef struct{

    uint32_t previous_size;
    uint32_t current_size;
    Grid* bias;
    Grid* weights;
    Grid* Before_Activation;
    Grid* After_Activation;
    double (*activation)(double);

}FullyConnected;



//In case we are dealing with a convolution / pooling
typedef struct{

        uint32_t size_kernel;
        char* choice;

}pool_information;

typedef union{

    Block* block;
    Grid* grid;

}Ker;

typedef union Kernels_{

    pool_information* psool;
    Grid* grid;
    Block* block;
    Blocks* blocks;

}Kernels;

typedef union data_{

    Grid* grid;
    Block* block;
    FullyConnected* fc;

}data;

typedef struct LAYER_{

    data* input_data;
    data* output_data;

    double (*activation__)(double);

    Kernels* kernels;
    Kernels* deltas;
    Kernels* cash;

    TYPE_LAYER name;

    struct LAYER_* next_layer;
    struct LAYER_* previous_layer;

}LAYER;


typedef struct {

    Block* X;
    Grid* Y;

    LAYER* first_layer;
    LAYER* final_layer;
    uint32_t nbr_levels;

}Model;


typedef struct node{
    uint32_t index;
    double value;

}node;

typedef struct pair
{
  node max_node;
  node min_node;
}pair;

pair getMinMax(Grid* grid, uint32_t low, uint32_t high)
{
  pair pminmax, pl,pr;
  uint32_t mid;


  if (low == high)
  {
     pminmax.max_node.value = grid->grid[low][0];
     pminmax.max_node.index = low;

     pminmax.min_node.value = grid->grid[low][0];
     pminmax.min_node.index = low;

     return pminmax;
  }


  if (high == low + 1)
  {
     if (grid->grid[low][0] > grid->grid[high][0])
     {


        pminmax.max_node.value = grid->grid[low][0];
        pminmax.max_node.index =low;

        pminmax.min_node.value = grid->grid[high][0];
        pminmax.min_node.index =high;


     }
     else
     {

        pminmax.max_node.value = grid->grid[high][0];
        pminmax.max_node.index =high;

        pminmax.min_node.value = grid->grid[low][0];
        pminmax.min_node.index =low;

     }
     return pminmax;
  }


  mid = (low + high)/2;
  pl = getMinMax(grid, low, mid);
  pr = getMinMax(grid, mid+1, high);


  if (pl.min_node.value < pr.min_node.value){

    pminmax.min_node.value = pl.min_node.value;
    pminmax.min_node.index=pl.min_node.index;
  }

  else{

    pminmax.min_node.value = pr.min_node.value;
    pminmax.min_node.index = pr.min_node.index;
  }


  if (pl.max_node.value < pr.max_node.value){

    pminmax.max_node.value = pr.max_node.value;
    pminmax.max_node.index =pr.max_node.index;
  }
  else{

    pminmax.max_node.value = pl.max_node.value;
    pminmax.max_node.index =pl.max_node.index;
  }

  return pminmax;
}



LAYER* initialize_LAYER(size_t size_allocation);
LAYER** initialize_pointer_LAYER(size_t size_allocation);
Model* initialize_Model(void);
Block* deep_block_copy(Block* block);
Grid* Flip_Grid(Grid* grid);
Block* Flip_Block(Block* block);

void display_Block(Block* block);
void display_Grid(Grid* grid);

void shape_block(Block* Block);
void shape_grid(Grid* grid);

void append_Block(Block** main_block, Block** block);

FullyConnected* deep_fc_copy(FullyConnected* fc);

unsigned int control_parity_kernel_size(unsigned int size_kernel);
void AddPadding_Block(Block** block, uint32_t padding);
int test_equal_grids_dimensions(Grid* grid1, Grid* grid2);
FullyConnected* initialize_Fully_Connected(size_t size_allocation);

double relu(double x);
double sigmoid(double x);
double cross_entropy_sample(Grid* y_hat, Grid* y);


void Convolution(Block** bl_output,
                 Block **input,
                 Blocks * kernels,
                 uint32_t stride,
                 uint32_t padding);

void Pooling(Block** bl_output,
             Block **input,
             Block **cash,
             uint32_t size_kernel,
             uint32_t stride,
             uint32_t padding,
             char* choice);

void Flatten(Block **output,
              Block **input);

void Fully_Connected_After_Flatten(FullyConnected** fc,
                                   Block** input,
                                   double (*activation)(double),
                                    int output_layer_size);

void Fully_Connected(FullyConnected** fc,
                     FullyConnected** fc_input,
                     double (*activation)(double),
                     int output_layer_size);

void Softmax_Activation(Grid** fc_output ,
                        FullyConnected** fc);

void create_Block(Block** block,
                  uint32_t input_depth,
                  uint32_t input_height,
                  uint32_t input_width,
                  char* choice,
                  char* type);

void create_Blocks(Blocks **blocks,
                   uint32_t length,
                   uint32_t depth,
                   uint32_t height,
                   uint32_t width,
                   char* choice,
                   char* type);


void initialize_layer_content_fc(LAYER** layer, FullyConnected** input){

    *layer=initialize_LAYER(1);

    (*layer)->input_data=(data*)malloc(sizeof(data));
    (*layer)->output_data=(data*)malloc(sizeof(data));

    (*layer)->input_data->fc=deep_fc_copy(*input);

    (*layer)->kernels=(Kernels*)malloc(sizeof(Kernels*));
    (*layer)->deltas=(Kernels*)malloc(sizeof(Kernels*));
    (*layer)->cash=(Kernels*)malloc(sizeof(Kernels*));

    (*layer)->kernels->blocks=NULL;
    (*layer)->kernels->block=NULL;
    (*layer)->kernels->psool=NULL;
    (*layer)->kernels->grid=NULL;

    //(*layer)->activation__=NULL;

    (*layer)->deltas->blocks=NULL;
    (*layer)->deltas->block=NULL;
    (*layer)->deltas->psool=NULL;
    (*layer)->deltas->grid=NULL;

    (*layer)->cash->blocks=NULL;
    (*layer)->cash->block=NULL;
    (*layer)->cash->psool=NULL;
    (*layer)->cash->grid=NULL;

    (*layer)->previous_layer=NULL;
    (*layer)->next_layer=NULL;

}

void initialize_layer_content_Block(LAYER** layer, Block** input){

    *layer=initialize_LAYER(1);

    (*layer)->input_data=(data*)malloc(sizeof(data));
    (*layer)->output_data=(data*)malloc(sizeof(data));

    (*layer)->input_data->block=deep_block_copy(*input);

    (*layer)->kernels=(Kernels*)malloc(sizeof(Kernels*));
    (*layer)->deltas=(Kernels*)malloc(sizeof(Kernels*));
    (*layer)->cash=(Kernels*)malloc(sizeof(Kernels*));

    (*layer)->kernels->blocks=NULL;
    (*layer)->kernels->block=NULL;
    (*layer)->kernels->psool=NULL;
    (*layer)->kernels->grid=NULL;

    //(*layer)->activation__=NULL;

    (*layer)->deltas->blocks=NULL;
    (*layer)->deltas->block=NULL;
    (*layer)->deltas->psool=NULL;
    (*layer)->deltas->grid=NULL;

    (*layer)->cash->blocks=NULL;
    (*layer)->cash->block=NULL;
    (*layer)->cash->psool=NULL;
    (*layer)->cash->grid=NULL;

    (*layer)->previous_layer=NULL;
    (*layer)->next_layer=NULL;

}

LAYER* conv_layer(paramsCONV prmconvs, Block* input){

    LAYER* layer;

    initialize_layer_content_Block(&layer,&input);

    Block* output;
    Blocks* kernels;
    Blocks* deltas;

    create_Blocks(&kernels,prmconvs.nbr_filters, input->depth,prmconvs.kernel_size,prmconvs.kernel_size,"random","float");
    create_Blocks(&deltas,prmconvs.nbr_filters, input->depth,prmconvs.kernel_size,prmconvs.kernel_size,"zeros","float");

    layer->kernels->blocks=kernels;
    layer->deltas->blocks=deltas;
    layer->input_data->block=input;

    Convolution(&output,
                &input,
                kernels,
                prmconvs.stride,
                prmconvs.padding);

    layer->cash->block=deep_block_copy(output);


    apply_function_to_Block(&output,
                            prmconvs.activation__
                            );

    layer->output_data->block=output;
    layer->activation__=prmconvs.activation__;
    layer->name=prmconvs.name;

    return layer;

    }



LAYER* Dense(FullyConnected* input){

    LAYER* layer;

    initialize_layer_content_fc(&layer,&input);

    FullyConnected* fc=initialize_Fully_Connected(1);
    current_Layer("Softmax Activation");

    Grid* fc_output;
    Softmax_Activation(&fc_output,&input);

    layer->output_data->grid=fc_output;
    layer->name=ACTIVATION__;

    return layer;

    }





LAYER* pool_layer(paramsPOOL prmpool, Block* input){

    LAYER* layer;

    initialize_layer_content_Block(&layer,&input);

    Block* output;
    Block* cash;

    layer->kernels->psool=(pool_information*)malloc(sizeof(pool_information));
    layer->kernels->psool->size_kernel=prmpool.kernel_size;


    Pooling(&output,&input,&cash,
                prmpool.kernel_size,
                prmpool.stride,
                prmpool.padding,
                prmpool.pooling_choice);


    //Add a block to recognize the elements

    write("---------------");
    write("--cash--");
    shape_block(cash);


    write("---------------");

    layer->input_data->block=input;
    layer->cash->block=cash;
    layer->output_data->block=output;
    layer->name=prmpool.name;

    return layer;

    }



LAYER* flatten_layer(paramsFLATTEN prmft, Block* input){

    LAYER* layer;

    initialize_layer_content_Block(&layer,&input);

    Block* output;
    Flatten(&output,&input);

    layer->input_data->block=input;
    layer->output_data->block=output;
    layer->name=prmft.name;

    return layer;

    }


LAYER* fcaf_layer(paramsFCAF prmfcaf, Block* input){

    LAYER* layer;

    initialize_layer_content_Block(&layer,&input);

    FullyConnected* output=initialize_Fully_Connected(1);;

    layer->input_data->block=input;
    Fully_Connected_After_Flatten(&output,
                                  &input,
                                  prmfcaf.activation__,
                                  prmfcaf.output_size);

    Grid* deltas;
    create_Grid(&deltas,output->weights->height,output->weights->width,"zeros","float");

    layer->activation__=prmfcaf.activation__;
    layer->deltas->grid=deltas;
    layer->kernels->grid=output->weights;
    layer->output_data->fc=output;
    layer->name=prmfcaf.name;

    return layer;

    }


LAYER* fc_layer(paramsFC prmffc, FullyConnected* input){

    LAYER* layer;

    initialize_layer_content_fc(&layer,&input);


    FullyConnected* output=initialize_Fully_Connected(1);;
    Fully_Connected(&output,
                    &input,
                    prmffc.activation__,
                    prmffc.output_size);

    Grid* deltas;
    create_Grid(&deltas,output->weights->height,output->weights->width,"zeros","float");
    layer->deltas->grid=deltas;

    layer->input_data->fc=input;
    layer->activation__=prmffc.activation__;
    layer->kernels->grid=output->weights;
    layer->output_data->fc=output;
    layer->name=prmffc.name;


    return layer;

    }



void create_Model(Model** model, Block* X, Grid *Y){

    *model=initialize_Model();
    (*model)->first_layer=NULL;
    (*model)->final_layer=NULL;
    (*model)->X=X;
    (*model)->Y=Y;
    (*model)->nbr_levels=0;

}

void determine_pointers_first_and_last(Model** model, LAYER** current_first, LAYER** current_last, LAYER** layer){

    if(!*current_first && ! *current_last){

        (*layer)->next_layer=NULL;
        (*layer)->previous_layer=NULL;

        *current_first=*layer;
        *current_last=*current_first;

    }
    else

    if(*current_first==*current_last){

        (*current_last)->next_layer=initialize_LAYER(1);
        (*current_last)->next_layer=*layer;
        (*layer)->previous_layer=*current_first;
        (*layer)->next_layer=NULL;
        *current_last=*layer;


    }else

    {
        while((*current_last)->next_layer)
            *current_last=(*current_last)->next_layer;

        (*current_last)->next_layer=initialize_LAYER(1);
        (*current_last)->next_layer=*layer;
        (*layer)->previous_layer=*current_last;
        (*layer)->next_layer=NULL;
        *current_last=*layer;

    }


}

void update_model(Model** model, LAYER** layer){

    LAYER* current_first=(*model)->first_layer;
    LAYER* current_last=(*model)->final_layer;

    determine_pointers_first_and_last(model, &current_first, &current_last, layer);

    (*model)->first_layer=current_first;
    (*model)->final_layer=current_last;


}


void add_CONV(Model** model, uint32_t nbr_filters,
                             uint32_t stride,
                             uint32_t padding,
                             uint32_t kernel_size,
                             double (*activation)(double)
                            ){

    paramsCONV params_conv={name:CONV,
                            stride:stride,
                            padding:padding,
                            nbr_filters: nbr_filters,
                            kernel_size:kernel_size,
                            activation__:activation,
                            };

    LAYER* conv_l=initialize_LAYER(1);

    (*model)->nbr_levels++;

    if(!(*model)->final_layer){
        conv_l=conv_layer(params_conv,(*model)->X);

    }
    else{

        conv_l=conv_layer(params_conv,(*model)->final_layer->output_data->block);

    }

    update_model(model,&conv_l);

    shape_block((*model)->final_layer->output_data->block);

}

void add_POOL(Model** model, uint32_t stride,
                             uint32_t padding,
                             uint32_t kernel_size,
                             char* pooling_choice){

    paramsPOOL params_pool={name:POOL,stride:stride,
                            padding:padding,
                            pooling_choice: pooling_choice,
                            kernel_size: kernel_size
                            };


    LAYER* pool_l=initialize_LAYER(1);
    (*model)->nbr_levels++;

    if((*model)->final_layer){

        pool_l=pool_layer(params_pool,(*model)->final_layer->output_data->block);
    }
    else{

        write("Cannot use Pooling layer at this level .. ");
        exit(0);

    }

    update_model(model,&pool_l);
    shape_block((*model)->final_layer->output_data->block);

}

void add_FLAT(Model** model){

    paramsFLATTEN params_fl={name:FLATTEN};

    LAYER* flatten_l=initialize_LAYER(1);
    (*model)->nbr_levels++;

    if((*model)->final_layer){

        flatten_l=flatten_layer(params_fl,(*model)->final_layer->output_data->block);
    }
    else{

        write("Cannot use Flattening layer at this level .. ");
        exit(0);

    }

    update_model(model,&flatten_l);
    shape_block((*model)->final_layer->output_data->block);

}

void add_FCAF(Model** model,double (*activation)(double),
                            uint32_t output_size){

    paramsFCAF params_fcaf={name:FULLY_CONNECTED_AFTER_FLATTEN,
                            activation__:activation,
                            output_size:output_size};


    LAYER* fcaf_l=initialize_LAYER(1);
    (*model)->nbr_levels++;

    if((*model)->final_layer){

        fcaf_l=fcaf_layer(params_fcaf,(*model)->final_layer->output_data->block);
    }
    else{

        write("Cannot use Fully Connected After Flattening layer at this level .. ");
        exit(0);

    }

    update_model(model,&fcaf_l);
    shape_grid((*model)->final_layer->output_data->grid);

}

void add_FC(Model** model,double (*activation)(double),
                            uint32_t output_size){

    paramsFC params_fc={name:FULLY_CONNECTED,
                            activation__:activation,
                            output_size:output_size};


    LAYER* fcaf_l=initialize_LAYER(1);
    (*model)->nbr_levels++;

    if((*model)->final_layer){

        fcaf_l=fc_layer(params_fc,(*model)->final_layer->output_data->fc);
    }
    else{

        write("Cannot use Fully Connected layer at this level .. ");
        exit(0);

    }

    update_model(model,&fcaf_l);
    shape_grid((*model)->final_layer->output_data->fc->After_Activation);

}


void DENSE(Model** model){


    LAYER* act_l=initialize_LAYER(1);
    (*model)->nbr_levels++;

    if((*model)->final_layer){

        act_l=Dense((*model)->final_layer->output_data->fc);
    }
    else{

        write("Cannot use Softmax Activation layer at this level .. ");
        exit(0);

    }

    update_model(model,&act_l);
    shape_grid((*model)->final_layer->output_data->fc);

}




double add__(double a, double b){
    return a+b;
}

double substract__(double a, double b){
    return a-b;
}

double multiply__(double a, double b){
    return a-b;
}




double relu(double x){
    return max(0,x);
}

double deriv_relu(double x){

    return 0*(x<=0)+1*(x>0);

}

double sigmoid(double x){

   return 1./(1.+exp(-x));
}

double deriv_sigmoid(double x){
    return sigmoid(x)*(1-sigmoid(x));
}

double deriv_tanh(double x){

    return 1-pow(tanh(x),2);
}


typedef struct{
    double (*add)(double,double);
    double (*substract)(double,double);
    double (*multiply)(double,double);
}Operator;

typedef struct{
    double (*function)(double);
    double (*prime)(double);

}Object;

Operator Op ={add:add__,substract:substract__,multiply:multiply__};

Object Tanh_Ojb={function:tanh, prime:deriv_tanh};
Object Sigm_Obj={function:sigmoid, prime:deriv_sigmoid};
Object Relu_Obj={function:relu, prime:deriv_relu};

Object function_to_object(double (*function)(double)){

    if(function==tanh) return Tanh_Ojb;
    if(function==sigmoid) return Sigm_Obj;
    if(function==relu) return Relu_Obj;

}

double generate_random(char* type){
    if(type=="int")
    return rand()-UPPER_BOUND/2;
    if(type="float")
    return ((double)rand())/((double)RAND_MAX) * UPPER_BOUND-UPPER_BOUND/2;
        else{

        printf("Uknown type .. exiting ..");
        exit(0);

        }
}

unsigned int control_parity_kernel_size(unsigned int size_kernel){
    return size_kernel%2==1;
}


unsigned int determine_size_output(int input_height,unsigned int kernel_height,unsigned int padding,unsigned int stride){

     return (int)(((input_height-kernel_height+2*padding)/stride))+1;
}




void shape_block(Block* block){

    printf("depth : %d \n",block->depth);
    printf("height : %d \n",block->height);
    printf("width : %d \n\n",block->width);

}

void shape_grid(Grid* grid){

    printf("height : %d \n",grid->height);
    printf("width : %d \n\n",grid->width);

}


unsigned int test_block_null_dimension(Block* block){

    return block->height && block->width && block->depth;
}

unsigned int test_grid_null_dimension(Grid* grid){

    return grid->height && grid->width ;
}


unsigned int test_block_for_fully_connected(Block* block){

    return block->width==1 && block->height==1 ;
}

unsigned int test_for_grid_elementwise_multiplication(Grid* grid1, Grid* grid2){

    return grid1->height==grid2->height && grid1->width==grid2->width ;

}

unsigned int test_equal_blocks_dimensions(Block* block1, Block* block2){

    return block1->depth==block2->depth && block1->width==block2->width &&\
                            block1->height==block2->height ;
}

unsigned int test_if_fully_connected_is_null(FullyConnected* fc){

    return fc->After_Activation!=NULL && fc->Before_Activation!=NULL;

}


double*** initialize_triple_pointer_double(size_t size_allocation){

    return malloc(size_allocation*sizeof(double**));
}


double** initialize_double_pointer_double(size_t size_allocation){

    return malloc(size_allocation*sizeof(double*));
}


double* initialize_pointer_double(size_t size_allocation){
    return malloc(size_allocation*sizeof(double));
}


LAYER** initialize_pointer_LAYER(size_t size_allocation){
    return malloc(size_allocation*sizeof(LAYER*));
}


LAYER* initialize_LAYER(size_t size_allocation){
    return malloc(size_allocation*sizeof(LAYER));
}

Model* initialize_Model(void){

    return malloc(sizeof(Model));

}


FullyConnected** initialize_pointer_Fully_Connected(size_t size_allocation){

    return malloc(size_allocation*sizeof(FullyConnected*));
}

FullyConnected* initialize_Fully_Connected(size_t size_allocation){

    return malloc(size_allocation*sizeof(FullyConnected));
}

Block* initialize_Block(size_t size_allocation){

    Block* block=malloc(size_allocation*sizeof(Block));

    block->depth=0;
    block->height=0;
    block->width=0;

}

Block** initialize_pointer_Block(size_t size_allocation){

    return malloc(size_allocation*sizeof(Block*));
}


Grid* initialize_Grid(size_t size_allocation){

    return malloc(size_allocation*sizeof(Grid));
}

Grid** initialize_pointer_Grid(size_t size_allocation){

    return malloc(size_allocation*sizeof(Grid*));
}



void create_Grid(Grid** grid,unsigned int input_height,unsigned int input_width,char* choice_content, char* type){

    *grid=(Grid*)malloc(sizeof(Grid));
    (*grid)->height=input_height;
    (*grid)->width=input_width;


    if(choice_content=="random"){

        (*grid)->grid=(double**)malloc(input_height*sizeof(double*));
       uint32_t counter_height;

        for(counter_height=0;counter_height<input_height;counter_height++){
           uint32_t counter_width;
            double* row=(double*)malloc(input_width*(sizeof(double)));

            for(counter_width=0;counter_width<input_width;counter_width++){
                        *(row+counter_width)=(double)generate_random(type);
                }
            *((*grid)->grid+counter_height)=row;
        }


    }

    else if(choice_content=="zeros"){

       (*grid)->grid=(double**)malloc(input_height*sizeof(double*));
       uint32_t counter_height;

        for(counter_height=0;counter_height<input_height;counter_height++){
           uint32_t counter_width;
            double* row=(double*)malloc(input_width*(sizeof(double)));

            for(counter_width=0;counter_width<input_width;counter_width++){
                        *(row+counter_width)=0.0;
                }
            *((*grid)->grid+counter_height)=row;
        }

    }

    else if(choice_content=="ones"){

       (*grid)->grid=(double**)malloc(input_height*sizeof(double*));
       uint32_t counter_height;

        for(counter_height=0;counter_height<input_height;counter_height++){
           uint32_t counter_width;
            double* row=(double*)malloc(input_width*(sizeof(double)));

            for(counter_width=0;counter_width<input_width;counter_width++){
                        *(row+counter_width)=1.0;
                }
            *((*grid)->grid+counter_height)=row;
        }

    }


}

Grid* transpose(Grid* grid_to_transpose){

    Grid* transposed_grid=(Grid*)malloc(sizeof(Grid));
    transposed_grid->height=grid_to_transpose->width;
    transposed_grid->width=grid_to_transpose->height;

    uint32_t index_heigth, index_width;
    transposed_grid->grid=(double**)malloc(transposed_grid->height*sizeof(double*));

    for(index_heigth=0;index_heigth<transposed_grid->height;index_heigth++){

        double* row=(double*)malloc(transposed_grid->width*sizeof(double));
        for(index_width=0;index_width<transposed_grid->width;index_width++){

            *(row+index_width)=grid_to_transpose->grid[index_width][index_heigth];
        }
        *(transposed_grid->grid+index_heigth)=row;
    }

    return transposed_grid;
}

double Sum_Grid(Grid* grid){

    double output=0.;
    uint32_t index_width, index_height;

   for(index_height=0;index_height<grid->height;index_height++){
            for(index_width=0;index_width<grid->width;index_width++){

                output+=grid->grid[index_height][index_width];

            }
   }

   return output;

}

void multiply_by_digit(Grid** grid, double digit){

    uint32_t index_width, index_height;

   for(index_height=0;index_height<(*grid)->height;index_height++){
            for(index_width=0;index_width<(*grid)->width;index_width++){

                (*grid)->grid[index_height][index_width]*=digit;

            }
   }

}

Grid* deep_grid_copy(Grid* grid){

    Grid* aux=initialize_Grid(1);
    aux->height=grid->height;
    aux->width=grid->width;
    aux->grid=initialize_double_pointer_double(aux->height);

    uint32_t row,col;

    for(row=0;row<aux->height;row++){
        double *row_ptr=(double*)malloc(aux->width*sizeof(double));
        for(col=0;col<aux->width;col++){
            *(row_ptr+col)=grid->grid[row][col];
            }
        *(aux->grid+row)=row_ptr;
    }

    return aux;

}




Grid* Operate(Grid* grid1, Grid* grid2, char* choice){

    if(!test_equal_grids_dimensions(grid1,grid2)){

        ERROR_DIMENSION_CONV;
        exit(0);

    }else
    {

        Grid* output=(Grid*)malloc(sizeof(Grid));
        output->height=grid1->height;
        output->width=grid1->width;
        output->grid=grid1->grid;

        uint32_t index_width, index_height;

        for(index_height=0;index_height<output->height;index_height++){
            for(index_width=0;index_width<output->width;index_width++){

                if(choice=="+")
                {
                output->grid[index_height][index_width]=Op.add(output->grid[index_height][index_width],\
                                                               grid2->grid[index_height][index_width]);
                }
                else
                if(choice=="-")
                {
                output->grid[index_height][index_width]=Op.substract(output->grid[index_height][index_width],\
                                                                grid2->grid[index_height][index_width]);
                }
                else
                if(choice=="*")
                {

                output->grid[index_height][index_width]=Op.multiply(output->grid[index_height][index_width],\
                                                                grid2->grid[index_height][index_width]);
                }
                else{
                    printf("Wrong operator choice :((( .. ");
                }
            }
        }

        return output;

    }
}


int test_for_grid_dot_multiplication(Grid* grid1, Grid* grid2){

    if(grid1->width==grid2->height){
        return 1;}

        else{

    if(transpose(grid1)->width==grid2->height){
        printf("Perhaps you should transpose the first grid ..");
    }
    if(grid1->width==transpose(grid2)->height){
        printf("Perhaps you should transpose the second grid ..");
    }
    if(transpose(grid1)->width==transpose(grid2)->height){
        printf("Perhaps you should transpose the two grids ..");
    }

    return 0;
    }

}

int test_equal_grids_dimensions(Grid* grid1, Grid* grid2){

    if(grid1->width==grid2->width && grid2->height==grid2->height)
    {
        return 1;
    }else
    {
    if(transpose(grid1)->width==grid2->width && transpose(grid1)->height==grid2->height){
        printf("Perhaps you should transpose the first grid ..");
    }

    if(transpose(grid2)->width==grid1->width && transpose(grid2)->height==grid1->height){
        printf("Perhaps you should transpose the second grid ..");
    }

    return 0;
    }
}


void create_Block(Block** block,unsigned int input_depth,unsigned int input_height,unsigned int input_width,char* choice, char* type){

    *block=(Block*)malloc(sizeof(Block));
    (*block)->height=input_height;
    (*block)->width=input_width;
    (*block)->depth=input_depth;

    (*block)->matrix=(double***)malloc(input_depth*sizeof(double**));

   uint32_t index_depth;
    for(index_depth=0;index_depth<input_depth;index_depth++){

        Grid* grid;
        create_Grid(&grid,input_height,input_width,choice,type);
        *((*block)->matrix+index_depth)=grid->grid;

    }

}


void create_Blocks(Blocks **blocks,unsigned int length,unsigned int depth,unsigned int height,unsigned int width, char* choice, char* type){

    if(length<1){

        ERROR_CREATING_BLOCKS;
        exit(0);

    }else{

       *blocks=(Blocks *)malloc(sizeof(Blocks));
       Blocks* blocks_tmp=*blocks;
       blocks_tmp->blocks=malloc(length*sizeof(Block));
      uint32_t index_length;

       for(index_length=0;index_length<length;index_length++){
            Block *new_block=(Block*)malloc(sizeof(Block));
            create_Block(&new_block,depth,height,width,choice,type);

            *(blocks_tmp->blocks+index_length)=new_block;
       }

       blocks_tmp->length=length;

    }
}

Grid* extract_grid_from_given_depth(Block** block, uint32_t index_depth){

    Grid* current_depth=(Grid*)malloc(sizeof(Grid));

    (current_depth)->height=(*block)->height;
    (current_depth)->width=(*block)->width;
    (current_depth)->grid=(*block)->matrix[index_depth];

    return current_depth;


}


Block* deep_block_copy(Block* block){

    Block* aux=initialize_Block(1);

    aux->height=block->height;
    aux->width=block->width;
    aux->depth=block->depth;

    aux->matrix=initialize_triple_pointer_double(aux->depth);

    unsigned depth;

    for(depth=0;depth<aux->depth;depth++){

            Grid* current_grid=extract_grid_from_given_depth(&block,depth);
            *(aux->matrix+depth)=current_grid->grid;

            free(current_grid);


    }

    return aux;

}

FullyConnected* deep_fc_copy(FullyConnected* fc){

    FullyConnected* aux=initialize_Fully_Connected(1);

    aux->activation=fc->activation;
    aux->After_Activation=deep_grid_copy(fc->After_Activation);
    aux->Before_Activation=deep_grid_copy(fc->Before_Activation);
    aux->bias=deep_grid_copy(fc->bias);
    aux->current_size=fc->current_size;
    aux->previous_size=fc->previous_size;
    aux->weights=deep_grid_copy(fc->weights);

    return aux;

}



void apply_function_to_Grid(Grid** grid, double (*pointer_to_function)(double)){

    int index_width,index_height;

    for(index_height=0;index_height<(*grid)->height;index_height++){
        for(index_width=0;index_width<(*grid)->width;index_width++){
                double *current_element=&((*grid)->grid[index_height][index_width]);
                ((*grid)->grid[index_height][index_width])=(*pointer_to_function)(*current_element);


        }
    }

}

void apply_function_to_Grid_softmax(Grid** grid, double (*pointer_to_function)(double)){

    uint32_t index_width,index_height;
    Grid* to_exponential=*grid;
    apply_function_to_Grid(&to_exponential,&exp);

    double sum_grid=Sum_Grid(to_exponential);

    for(index_height=0;index_height<(*grid)->height;index_height++){
        for(index_width=0;index_width<(*grid)->width;index_width++){

                double *current_element=&((*grid)->grid[index_height][index_width]);
                ((*grid)->grid[index_height][index_width])=(*current_element)/sum_grid;

        }
    }

}



void apply_function_to_Block(Block** block, double (*pointer_to_function)(double)){

    int index_depth;
    int height=(*block)->height;
    int width=(*block)->width;

    for(index_depth=0;index_depth<(*block)->depth;index_depth++){

            Grid* current_depth=extract_grid_from_given_depth(block,index_depth);

            apply_function_to_Grid(&current_depth,pointer_to_function);

            (*block)->matrix[index_depth]=current_depth->grid;

            free(current_depth);

    }



}

//Extract a smaller Grid From a Grid
Grid* Extract_From_Grid(Grid* grid,\
                         uint32_t begin_input_height,unsigned int end_input_height,\
                         uint32_t begin_input_width,unsigned int end_input_width){

    Grid* output_grid=(Grid*)malloc(sizeof(Grid));

    uint32_t size_height=end_input_height-begin_input_height;
    uint32_t size_width=end_input_width-begin_input_width;

    output_grid->width=size_width;
    output_grid->height=size_height;

    output_grid->grid=(double**)malloc(output_grid->height*sizeof(double*));

    uint32_t counter_height;
    uint32_t new_counter_height=0;

    for(counter_height=begin_input_height;counter_height<end_input_height;counter_height++){

        uint32_t counter_width;
        uint32_t new_counter_width=0;
        double* row=(double*)malloc(output_grid->width*(sizeof(double)));

        for(counter_width=begin_input_width;counter_width<end_input_width;counter_width++){
                    *(row+new_counter_width)=grid->grid[counter_height][counter_width];
                    new_counter_width++;

            }

        output_grid->grid[new_counter_height]=row;
        new_counter_height++;
    }

    return output_grid;

}


Block* Extract_From_Block(Block* grid,
                         uint32_t begin_input_depth,unsigned int end_input_depth,
                         uint32_t begin_input_height,unsigned int end_input_height,
                         uint32_t begin_input_width,unsigned int end_input_width){


    Block* output_grid=(Block*)malloc(sizeof(Block));

    uint32_t size_depth=end_input_depth-begin_input_depth;
    uint32_t size_height=end_input_height-begin_input_height;
    uint32_t size_width=end_input_width-begin_input_width;

    output_grid->depth=size_depth;
    output_grid->width=size_width;
    output_grid->height=size_height;

    output_grid->matrix=(double***)malloc(size_depth*sizeof(double**));

    uint32_t counter_depth;
    uint32_t new_counter_depth=0;

    for(counter_depth=begin_input_depth;counter_depth<end_input_depth;counter_depth++){

                    Grid* grid_from_current_block=(Grid*)malloc(sizeof(Grid));
                    grid_from_current_block->height=size_height;
                    grid_from_current_block->width=size_width;
                    grid_from_current_block->grid=grid->matrix[counter_depth];

                    Grid* extracted_grid=Extract_From_Grid(grid_from_current_block,begin_input_height,end_input_height,\
                          begin_input_width,end_input_width);
                        *((output_grid->matrix)+new_counter_depth)=extracted_grid->grid;
                        new_counter_depth++;

                    free(grid_from_current_block);

                }

    return output_grid;

}


Grid* AddPadding_Grid(Grid** block, uint32_t padding){

    Grid* output_grid;

    uint32_t height=(*block)->height;
    uint32_t width=(*block)->width;
    double **block_matrix=(*block)->grid;

    create_Grid(&output_grid,height+2*padding,width+2*padding,"zeros","float");

    //double** output_grid_matrix=output_grid->grid;

    uint32_t counter_height;
    uint32_t counter_width;


        for(counter_height=padding;counter_height<padding+height;counter_height++){

            for(counter_width=padding;counter_width<padding+width;counter_width++){

                output_grid->grid[counter_height][counter_width]=\
                    block_matrix[counter_height-padding][counter_width-padding];

            }
    }

    return output_grid;

}

Grid* Flip_Grid(Grid* grid){

    Grid* flipped_grid=initialize_Grid(1);
    flipped_grid->width=grid->width;
    flipped_grid->height=grid->height;

    flipped_grid->grid=initialize_double_pointer_double(flipped_grid->height);

    uint32_t index_width, index_height;
    for(index_height=0;index_height<flipped_grid->height;index_height++){

        double* row=initialize_pointer_double(flipped_grid->width);
        for(index_width=0;index_width<flipped_grid->width;index_width++){

            *(row+index_width)=grid->grid[flipped_grid->height-index_height-1][flipped_grid->width-index_width-1];
        }

        *(flipped_grid->grid+index_height)=row;
    }


    return flipped_grid;

}


Block* Flip_Block(Block* block){

    Block* flipped=initialize_Block(1);
    flipped->depth=block->depth;
    flipped->height=block->height;
    flipped->width=block->width;

    flipped->matrix=initialize_triple_pointer_double(flipped->depth);

    uint32_t index_depth;

    for(index_depth=0;index_depth<flipped->depth;index_depth++){

        Grid* grid=extract_grid_from_given_depth(&block,index_depth);
        Grid* flipped_grid=Flip_Grid(grid);
        *(flipped->matrix+index_depth)=flipped_grid->grid;

        free(grid);
        free(flipped_grid);

    }

    return flipped;

}



void AddPadding_Block(Block** block, uint32_t padding){

    Block *output_Block=(Block*)malloc(sizeof(Block));

    output_Block->depth=(*block)->depth;

    output_Block->height=(*block)->height+2*padding;
    output_Block->width=(*block)->width+2*padding;
    output_Block->matrix=(double***)malloc(output_Block->depth*sizeof(double**));


   uint32_t counter_depth;

    for(counter_depth=0;counter_depth<output_Block->depth;counter_depth++){

            Grid* padded_grid=extract_grid_from_given_depth(block,counter_depth);

            *(output_Block->matrix+counter_depth)=AddPadding_Grid(&padded_grid,padding)->grid;

            free(padded_grid);


    }

    *block=output_Block;

}

double convolve_multiplication_sum(Block* block1, Block* block2){
   uint32_t depth,width,height;

    if(!test_equal_blocks_dimensions(block1,block2)){

                ERROR_DIMENSION_CONV;
                exit(0);
                            }
    else{
        double output=0;

        for(depth=0;depth<block1->depth;depth++){

            for(height=0;height<block1->height;height++){

                for(width=0;width<block1->width;width++){
                    output+=block1->matrix[depth][height][width]*(block2->matrix[depth][height][width]);

            }
        }
    }


        return output;
    }
}



Grid* convolve(Block* block, Block* kernel,unsigned int stride,unsigned int padding){

    //DEBUG;
    if(test_block_null_dimension(block) && test_block_null_dimension(kernel)){

        if(!control_parity_kernel_size(kernel->height) || !control_parity_kernel_size(kernel->width)){

            ERROR_EVEN_DIMENSIONS;
            exit(0);
        }else

        if(block==NULL || kernel==NULL){

            ERROR_NULL;
            exit(0);

        }
        else
        if(block->depth!=kernel->depth){

            ERROR_DEPTH;
            exit(0);

        }
        else{

        uint32_t height=block->height;

        AddPadding_Block(&block,padding);

        uint32_t size_half_kernel=((kernel->height-1)/2);
        uint32_t begin_point_height=size_half_kernel;
        uint32_t end_point_height=block->height-begin_point_height;

        uint32_t begin_point_width=size_half_kernel;
        uint32_t end_point_width=block->width-begin_point_width;

        uint32_t index_height_output;
        uint32_t index_width_output;

        Grid* output_convolution_grid=(Grid*)malloc(sizeof(Grid));
        output_convolution_grid->height=(int)(end_point_height-begin_point_height)/stride;
        output_convolution_grid->width=(int)(end_point_width-begin_point_width)/stride;
        if(block->height%2) output_convolution_grid->height++ ;
        if(block->width%2) output_convolution_grid->width++ ;

        double** grid=(double**)malloc(output_convolution_grid->height*sizeof(double*));


        for(index_height_output=begin_point_height;index_height_output<end_point_height;index_height_output+=stride){

            double *row=(double*)malloc(output_convolution_grid->width*sizeof(double));

            for(index_width_output=begin_point_width;index_width_output<end_point_width;index_width_output+=stride){


                Block* extracted_block=Extract_From_Block(block,0,kernel->depth,index_height_output-size_half_kernel,\
                                                          index_height_output+size_half_kernel+1,index_width_output-size_half_kernel,\
                                                          index_width_output+size_half_kernel+1);

                //write("yes");

                *(row+(index_width_output-begin_point_width)/stride)=convolve_multiplication_sum(extracted_block,kernel);

            }
            *(grid+(index_height_output-begin_point_height)/stride)=row;
        }


        output_convolution_grid->grid=grid;




        return output_convolution_grid;
       }
        }

    else{
        ERROR_NULL;
        exit(0);

    }
}


//Can either be used to define the input images or the N * filters

void Convolution(Block** bl_output, Block **input, Blocks * kernels,unsigned int stride,unsigned int padding){

    current_Layer("Convolution");

    if(!test_block_null_dimension(*input)){
        ERROR_NULL;
        exit(0);
    }
    else
    {

        Block* output=(Block*)malloc(sizeof(Block));
        output->depth=kernels->length;
        output->height=determine_size_output((*input)->height,kernels->blocks[0]->height, padding, stride);
        output->width=determine_size_output((*input)->width,kernels->blocks[0]->width, padding, stride);
        output->matrix=(double***)malloc(output->depth*sizeof(double**));


        // We have now to fill the output_matrix;

        uint32_t index_output_depth;

        for(index_output_depth=0;index_output_depth<output->depth;index_output_depth++){


            Grid* grid=convolve(*input, kernels->blocks[index_output_depth],stride,padding);
            *(output->matrix+index_output_depth)=grid->grid;
            free(grid);

        }

        *bl_output=output;
    }

}

typedef struct{

    double value;
    uint32_t index_height;
    uint32_t index_width;

}Entity;

typedef struct{
    Grid* special_grid;
    Grid* pooled;

}POOL_OUTPUT;

Entity* Pooling_On_Extracted_Grid(Grid* block, char* choice){

   Entity* ent=(Entity*)malloc(sizeof(Entity));

    if(choice=="max"){
        double output=0;

        uint32_t width,height;

        uint32_t index_height=0;
        uint32_t index_width=0;

        for(height=0;height<block->height;height++){
            for(width=0;width<block->width;width++){


                if(output<block->grid[height][width]){
                    output=block->grid[height][width];
                    index_height=height;
                    index_width=width;
                }

            }
        }

        ent->value=output;
        ent->index_height=index_height;
        ent->index_width=index_width;

        return ent;

    }else
    if(choice=="average"){

        //// Let us see
        uint32_t width,height;

        double output=0;

        for(height=0;height<block->height;height++){
            for(width=0;width<block->width;width++){

                    output+=block->grid[height][width];

            }
        }

        ent->value=output/((block->height)*(block->width));
        ent->index_height=-1;
        ent->index_width=-1;

        return ent;

        }
    }


POOL_OUTPUT* Pooling_On_Grid(Grid* grid,unsigned int size_kernel,unsigned int stride,unsigned int padding,char* choice){

    if(!test_grid_null_dimension(grid)){
        ERROR_NULL;
        exit(0);

    }else

    if(!control_parity_kernel_size(size_kernel)){
        ERROR_EVEN_DIMENSIONS;
        exit(0);

    }
    else {


   POOL_OUTPUT* po=(POOL_OUTPUT*)malloc(sizeof(POOL_OUTPUT));
   uint32_t height=grid->height;

   grid=AddPadding_Grid(&grid,padding);
   Grid* special_grid=NULL;

   if(choice=="avg"){

       create_Grid(&special_grid,grid->height,grid->width,"ones","float");
       multiply_by_digit(&special_grid,1/(special_grid->height*special_grid->width));

       }

    // We might as well add a size_output for the width

    uint32_t size_half_kernel=((size_kernel-1)/2);
    uint32_t begin_point_height=size_half_kernel;
    uint32_t end_point_height=grid->height-begin_point_height;

    uint32_t begin_point_width=size_half_kernel;
    uint32_t end_point_width=grid->width-begin_point_width;

    uint32_t index_height_output;
    uint32_t index_width_output;

    Grid* output_pooled_grid=(Grid*)malloc(sizeof(Grid));
    output_pooled_grid->height=(end_point_height-begin_point_height)/stride+1;
    output_pooled_grid->width=(end_point_width-begin_point_width)/stride+1;

    output_pooled_grid->grid=(double**)malloc(output_pooled_grid->height*sizeof(double*));

    if(special_grid==NULL)
        create_Grid(&special_grid,grid->height,grid->width,"zeros","float");


    for(index_height_output=begin_point_height;index_height_output<end_point_height;index_height_output+=stride){


        double *row=(double*)malloc(output_pooled_grid->width*sizeof(double));

        for(index_width_output=begin_point_width;index_width_output<end_point_width;index_width_output+=stride){

        Grid* extracted_grid=Extract_From_Grid(grid,index_height_output-size_half_kernel,\
                                                  index_height_output+size_half_kernel+1,index_width_output-size_half_kernel,\
                                                  index_width_output+size_half_kernel+1);

        Entity* ent=Pooling_On_Extracted_Grid(extracted_grid,choice);
        *(row+(index_width_output-begin_point_width)/stride)=0.0;

        if(choice=="max"){

            *(row+(index_width_output-begin_point_width)/stride)=ent->value;

            special_grid->grid[index_height_output-size_half_kernel+ent->index_height]
                          [index_width_output-size_half_kernel+ent->index_width]
                           = 1.0 ;
        }
        else{

            *(row+(index_width_output-begin_point_width)/stride)=ent->value;

        }


        }
        *(output_pooled_grid->grid+(index_height_output-begin_point_height)/stride)=row;



    }


    po->pooled=output_pooled_grid;
    po->special_grid=special_grid;

    return po;

    }
}


// We will continue at this level
void Pooling(Block** bl_output,Block **input,Block** cash, uint32_t size_kernel,unsigned int stride,unsigned int padding, char* choice){

    current_Layer("Pooling");

    if(!test_block_null_dimension(*input)){
        ERROR_NULL;
        exit(0);
    }
    else
    {

    Block* output=(Block*)malloc(sizeof(Block));
    Block* cash_bis=(Block*)malloc(sizeof(Block));

    output->height=determine_size_output((*input)->height, size_kernel, padding, stride);
    output->width=determine_size_output((*input)->width, size_kernel, padding, stride);

    cash_bis->height=(*input)->height;
    cash_bis->width=(*input)->width;

    output->depth=(*input)->depth;
    cash_bis->depth=(*input)->depth;

    output->matrix=(double***)malloc(output->depth*sizeof(double**));
    cash_bis->matrix=(double***)malloc(cash_bis->depth*sizeof(double**));

    // We have now to fill the output_matrix;



    uint32_t index_output_depth;

    for(index_output_depth=0;index_output_depth<output->depth;index_output_depth++){


        Grid* grid_from_current_block=extract_grid_from_given_depth(input,index_output_depth);
        POOL_OUTPUT* pooled_po=Pooling_On_Grid(grid_from_current_block,size_kernel,stride,padding,choice);

        *(output->matrix+index_output_depth)=pooled_po->pooled->grid;
        *(cash_bis->matrix+index_output_depth)=pooled_po->special_grid->grid;



       free(grid_from_current_block);
       free(pooled_po);


    }



    *bl_output=output;
    *cash=cash_bis;

    }
}


// Take a block and decrease both its width and height to 1
void extract_Grid_From_Flatten_Block(Block** block, Grid** grid){

    *grid=(Grid*)malloc(sizeof(Grid));
    (*grid)->height=(*block)->height;
    (*grid)->width=(*block)->depth;

    (*grid)->grid=(double**)malloc(sizeof(double*));
    double* row=(double*)malloc((*grid)->width*sizeof(double));

    uint32_t index_depth;
    for(index_depth=0;index_depth<(*grid)->width;index_depth++){

        *(row+index_depth)=(*block)->matrix[index_depth][0][0];
    }

    *((*grid)->grid)=row;

}


void get_Block_from_Grid(Grid** grid, Block** block){

    *block=initialize_Block(1);

    (*block)->depth=1;
    (*block)->height=(*grid)->height;
    (*block)->width=(*grid)->width;

    (*block)->matrix=initialize_triple_pointer_double((*block)->depth);

    *((*block)->matrix)=initialize_double_pointer_double((*grid)->height);
    *((*block)->matrix)=(*grid)->grid;

}

void extract_Flatten_Block_from_Grid(Grid** grid, Block** block){

    *block=initialize_Block(1);

    (*block)->depth=(*grid)->height;
    (*block)->height=1;
    (*block)->width=1;

    (*block)->matrix=initialize_triple_pointer_double((*block)->depth);

    uint32_t index_depth;
    for(index_depth=0;index_depth<(*block)->depth;index_depth++){

        *((*block)->matrix+index_depth)=initialize_double_pointer_double(1);
        **((*block)->matrix+index_depth)=initialize_pointer_double(1);
        ***((*block)->matrix+index_depth)=(*grid)->grid[index_depth][0];

    }


}



void Flatten(Block **output, Block **input){


    current_Layer("Flatten");

    if(!test_block_null_dimension(*input)){

        ERROR_NULL;
        exit(0);

    }else
    {

    Block* block=(Block*)malloc(sizeof(Block));

    block->depth=(*input)->depth*(*input)->height*(*input)->width;
    block->width=1;
    block->height=1;

    double*** Flattened=(double***)malloc(block->depth*sizeof(double**));

    uint32_t index_depth;
    uint32_t index_height;
    uint32_t index_width;

    uint32_t counter_array_flattened=0;

    for(index_depth=0;index_depth<(*input)->depth;index_depth++){

            for(index_height=0;index_height<(*input)->height;index_height++){

                for(index_width=0;index_width<(*input)->width;index_width++){

                    double** decrease_height=(double**)malloc(sizeof(double*));
                    double* decrease_width=(double*)malloc(sizeof(double));

                    *decrease_width=(*input)->matrix[index_depth][index_height][index_width];
                    *decrease_height=decrease_width;


                    Flattened[counter_array_flattened]=decrease_height;

                    counter_array_flattened++;

            }

        }

    }

    block->matrix=Flattened;
    *output=block;

    }

}

void Stack_Blocks(int nbr_arguments,... ){

   uint32_t i;
    va_list argptr;
    /* initialize argptr for nbr_arguments number of arguments */
    va_start(argptr, nbr_arguments);


    for (i = 0; i < nbr_arguments; i++) {
      shape_block(va_arg(argptr, Block*));
      //va_arg(valist, Block*);
      /*

      Add other functions

      */
   }
   /* clean memory reserved for argptr */
   va_end(argptr);

}

void grid_dot_mutiplication(Grid** output_grid, Grid** grid1, Grid** grid2){

    if(!test_for_grid_dot_multiplication(*grid1,*grid2)){

        ERROR_DIMENSION_GRID_MULT;
        exit(0);

    }
    else{



        *output_grid=(Grid*)malloc(sizeof(Grid));
        (*output_grid)->height=(*grid1)->height;
        (*output_grid)->width=(*grid2)->width;

        (*output_grid)->grid=(double**)malloc((*output_grid)->height*sizeof(double*));


        int index_height, index_width;

        for(index_height=0;index_height<(*output_grid)->height;index_height++){


            double* row=(double*)malloc((*output_grid)->height*sizeof(double));

            for(index_width=0;index_width<(*output_grid)->width;index_width++){


                int index_forgotten;
                double sum_forgotten=0.0;

                for(index_forgotten=0;index_forgotten<(*grid1)->width;index_forgotten++){
                    sum_forgotten+=(*grid1)->grid[index_height][index_forgotten]*((*grid2)->grid[index_forgotten][index_width]);

                }
                *(row+index_width)=sum_forgotten;

            }
            *((*output_grid)->grid+index_height)=row;

            }
        }

}


void Fully_Connected_After_Flatten(FullyConnected** fc, Block** input, double (*activation)(double), int output_layer_size){

    current_Layer("Fully Connected");

    uint32_t input_layer_size=(*input)->depth;

    Grid* weights_tmp;
    Grid* Z_i;
    Grid* A_i;
    Grid* input_grid;

    if(!test_if_fully_connected_is_null(*fc)){

        if(!test_block_for_fully_connected(*input)){

            ERROR_DIM_FLATTEN ;
            exit(0);

        }else

        {

            *fc=(FullyConnected*)malloc(sizeof(FullyConnected));
            FullyConnected* local_fc=*fc;


            local_fc->bias=(Grid*)malloc(sizeof(Grid*));
            create_Grid(&local_fc->bias,output_layer_size,1,"zeros","float");

            Grid* weights_tmp;
            create_Grid(&weights_tmp,output_layer_size,input_layer_size,"random","float");

            local_fc->weights=weights_tmp;
            local_fc->activation=*activation;



            extract_Grid_From_Flatten_Block(input,&input_grid);
            Grid* transposed_input_grid=transpose(input_grid);

            grid_dot_mutiplication(&Z_i,&local_fc->weights,&transposed_input_grid);

            A_i=deep_grid_copy(Z_i);

            local_fc->Before_Activation=Z_i;

            Grid* A_i_plus_bias=Operate(A_i,local_fc->bias,"+");
            apply_function_to_Grid(&A_i_plus_bias,local_fc->activation);

            shape_grid(A_i_plus_bias);

            local_fc->After_Activation=A_i_plus_bias;
            local_fc->previous_size=input_layer_size;
            local_fc->current_size=output_layer_size;


            }
        }

    else{

            FullyConnected* local_fc=*fc;

            free(local_fc->Before_Activation);
            free(local_fc->After_Activation);

            extract_Grid_From_Flatten_Block(input,&input_grid);
            Grid* transposed_input_grid=transpose(input_grid);

            grid_dot_mutiplication(&Z_i,&local_fc->weights,&transposed_input_grid);

            A_i=deep_grid_copy(Z_i);

            local_fc->Before_Activation=Z_i;

            Grid* A_i_plus_bias=Operate(A_i,local_fc->bias,"+");
            apply_function_to_Grid(&A_i_plus_bias,local_fc->activation);

            local_fc->After_Activation=A_i_plus_bias;

            shape_grid(local_fc->After_Activation);

    }
}


void Fully_Connected(FullyConnected** fc, FullyConnected** fc_input,double (*activation)(double), int output_layer_size){

    current_Layer("Fully Connected");

    uint32_t input_layer_size=(*fc_input)->current_size;
    Grid* weights_tmp;
    Grid* Z_i;
    Grid* A_i;

    if(!test_if_fully_connected_is_null(*fc)){

        //test if a FullyConnected block is null
        if(!test_if_fully_connected_is_null(*fc_input)){

            ERROR_NULL ;
            exit(0);

        }else

        {

            *fc=(FullyConnected*)malloc(sizeof(FullyConnected));
            FullyConnected* local_fc=*fc;


            local_fc->bias=(Grid*)malloc(sizeof(Grid*));
            create_Grid(&local_fc->bias,output_layer_size,1,"zeros","float");


            create_Grid(&weights_tmp,output_layer_size,input_layer_size,"random","float");

            local_fc->weights=weights_tmp;
            local_fc->activation=*activation;

            grid_dot_mutiplication(&Z_i,&local_fc->weights,&(*fc_input)->After_Activation);
            //grid_dot_mutiplication(&A_i,&local_fc->weights,&(*fc_input)->After_Activation);

            A_i=deep_grid_copy(Z_i);

            local_fc->Before_Activation=Z_i;

            Grid* A_i_plus_bias=Operate(A_i,local_fc->bias,"+");
            apply_function_to_Grid(&A_i_plus_bias,local_fc->activation);

            local_fc->After_Activation=A_i_plus_bias;
            local_fc->previous_size=input_layer_size;
            local_fc->current_size=output_layer_size;

            shape_grid(local_fc->After_Activation);

        }

    }
    else{

            FullyConnected* local_fc=*fc;

            grid_dot_mutiplication(&Z_i,&local_fc->weights,&(*fc_input)->After_Activation);
            //grid_dot_mutiplication(&A_i,&local_fc->weights,&(*fc_input)->After_Activation);

            A_i=deep_grid_copy(Z_i);


            Grid* A_i_plus_bias=Operate(A_i,local_fc->bias,"+");
            apply_function_to_Grid(&A_i_plus_bias,local_fc->activation);

            local_fc->After_Activation=A_i_plus_bias;
            local_fc->previous_size=input_layer_size;
            local_fc->current_size=output_layer_size;

            shape_grid(local_fc->After_Activation);

    }
}


double cross_entropy_sample(Grid* y_hat, Grid* y){

    if(y_hat->height!=y->height){

        printf("final layer must have the same true output size ..");
        exit(0);
    }

    pair y_pair = getMinMax(y, 0, y->height-1);

    uint32_t wanted_index=y_pair.max_node.index;

    return -log(y_hat->grid[wanted_index][0]);

    }


Grid* fill_index(unsigned int height, uint32_t index){

    if(index>=height){
        printf("\nWrong index .. \n");
        exit(0);
    }

    Grid* output;
    create_Grid(&output,height,1,"zeros","float");
    output->grid[index][0]=1;
    return output;

}


Block* to_categorical(Grid* input){

    uint32_t depth=input->height;
    uint32_t index_depth;

    pair input_pair=getMinMax(input,0,input->height-1);

    Block* output=initialize_Block(1);
    output->depth=depth;
    output->height=(unsigned int)input_pair.max_node.value;
    output->width=1;

    output->matrix=(double***)malloc(output->depth*sizeof(double**));

    for(index_depth=0;index_depth<depth;index_depth++){

            *(output->matrix+index_depth)=fill_index(depth,(unsigned int)input->grid[index_depth][0])->grid;

    }

    return output;

}

void Softmax_Activation(Grid** fc_output ,FullyConnected** fc){

    *fc_output=deep_grid_copy((*fc)->After_Activation);
    apply_function_to_Grid_softmax(fc_output,&exp);

}


void display_Block(Block* grid){

   uint32_t dpth,row,col;

    for(dpth=0;dpth<grid->depth;dpth++){
        printf("Level : %d\n",dpth+1);
        for(row=0;row<grid->height;row++){
            for(col=0;col<grid->width;col++){
                printf("%.10lf |", grid->matrix[dpth][row][col]);
            }
            printf("\n");
        }
        printf(" \n \n");
    }
    printf("\n");
}


void display_Grid(Grid *table){
   uint32_t row,col;

    for(row=0;row<table->height;row++){
        for(col=0;col<table->width;col++){
            printf("%.10lf |", table->grid[row][col]);
            }
        printf("\n");
    }

    printf("\n");

}


void calculate_deltas_fc(Model** model, LAYER** layer){

    write(getType((*layer)->name));
    if((*layer)->name!=FULLY_CONNECTED_AFTER_FLATTEN &&\
            (*layer)->name!=FULLY_CONNECTED &&\
            (*layer)->name!=ACTIVATION__ &&\
            (*layer)->name!=FLATTEN
       ){

                printf("\nWrong use of functions ..\n");
                exit(0);

            }

    else
        {

            if((*layer)==(*model)->final_layer){

                Grid* final_delta=Operate((*layer)->output_data->grid,(*model)->Y,"-");
                (*layer)->deltas->grid=final_delta;


            }

            else

            if((*layer)==(*model)->final_layer->previous_layer){

                    (*layer)->deltas->grid=(*layer)->next_layer->deltas->grid;

            }

            else
            {

                    Grid* pre_deltas_output;
                    Grid* transposed_weights=transpose((*layer)->next_layer->kernels->grid);
                    grid_dot_mutiplication(&pre_deltas_output,&transposed_weights,&(*layer)->next_layer->deltas->grid);


                    if((*layer)->name==FLATTEN){

                        Block* data_to_flat=deep_block_copy((*layer)->output_data->block);

                        Grid* data_flattend;
                        extract_Grid_From_Flatten_Block(&data_to_flat,&data_flattend);

                        Grid* Z_previous=deep_grid_copy(data_flattend);

                        Object function=function_to_object((*layer)->next_layer->activation__);
                        apply_function_to_Grid(&Z_previous,function.prime);

                        Grid* deltas_output=Operate(pre_deltas_output,transpose(Z_previous),"*");

                        Block* final_deltas;

                        extract_Flatten_Block_from_Grid(&deltas_output,&final_deltas);
                        (*layer)->deltas->block=final_deltas;

                    }

                    else {

                        Grid* Z_previous=deep_grid_copy((*layer)->output_data->fc->Before_Activation);

                        Object function=function_to_object((*layer)->next_layer->activation__);
                        apply_function_to_Grid(&Z_previous,function.prime);

                        Grid* deltas_output=Operate(pre_deltas_output,Z_previous,"*");

                        (*layer)->deltas->grid=deltas_output;

                    }


                    /*
                    write("--------------------------");
                    write("Deltas+1");
                    shape_grid((*layer)->next_layer->deltas->grid);

                    write("A_I");
                    shape_grid(transpose((*layer)->output_data->fc->After_Activation));

                    write("matching this layer..");

                    shape_grid((*layer)->kernels->grid);

                    write("or next layer ..");

                    shape_grid((*layer)->next_layer->kernels->grid);
                    write("--------------------------");
                    */

                    //Next thing .. we will update the next layer for sure :)))


            }

        }
}

void calculate_deltas__pool(Model** model, LAYER** layer){

    if((*layer)->name!=POOL)
        {

                printf("\nWrong use of functions ..\n");
                exit(0);

        }

    else{

        if((*layer)->next_layer->name==FLATTEN){

            Block* deltas=(*layer)->next_layer->deltas->block;
            Block* block_for_dimensions=(*layer)->next_layer->input_data->block;

            Block* current_deltas=initialize_Block(1);

            current_deltas->width=block_for_dimensions->width;
            current_deltas->depth=block_for_dimensions->depth;
            current_deltas->height=block_for_dimensions->height;

            uint32_t ind_d,ind_h,ind_w,current=0;

            current_deltas->matrix=initialize_triple_pointer_double(current_deltas->depth);

            for(ind_d=0;ind_d<current_deltas->depth;ind_d++){
                double** col=initialize_double_pointer_double(current_deltas->height);

                for(ind_h=0;ind_h<current_deltas->height;ind_h++){
                    double* row=initialize_pointer_double(current_deltas->width);

                    for(ind_w=0;ind_w<current_deltas->width;ind_w++){

                        *(row+ind_w)=deltas->matrix[current][0][0];
                        current++;
                    }
                    *(col+ind_h)=row;
                }
                *(current_deltas->matrix+ind_d)=col;
            }

            shape_block(current_deltas);
            (*layer)->deltas->block=current_deltas;
        }

    }
    //go through each delta output and reconstruct a decent delta input
}

Block* extract_from_Blocks(Blocks* blocks, uint32_t index){

    return blocks->blocks[index];
}


Block* get_one_dimension_from_block(Block* block, uint32_t depth){

    Grid* extracted_grid=extract_grid_from_given_depth(&block, depth);
    Block* block_;
    get_Block_from_Grid(&extracted_grid,&block_);

    return block_;

}

Block* element_wise_multiplication(Block* block1, Block* block2){

        /*
        if(test_equal_blocks_dimensions(block1,block2)){

            ERROR_DIM_BLOCK_DIMENSIONS;
            exit(0);

        }

        else{
            */

            Block* product=initialize_Block(1);

            product->depth=block1->depth;
            product->height=block1->height;
            product->width=block1->width;

            uint32_t index_d;

            for(index_d=0;index_d<(block1->depth);index_d++){

                Grid* first_grid=extract_grid_from_given_depth(&block1,index_d);
                Grid* second_grid=extract_grid_from_given_depth(&block2,index_d);

                Grid* result=Operate(first_grid,second_grid,"*");

                Block* output;

                get_Block_from_Grid(&result,&output);

                display_Block(output);

                append_Block(&product,&output);
                write("ok");
            }

            return product;

        //}

}

Block* crop_Block(Block* to_be_cropped, Block* to_be_used){

    Block* cropped=initialize_Block(1);

    cropped->depth=to_be_cropped->depth;
    cropped->height=to_be_used->height;
    cropped->width=to_be_used->width;

    cropped->matrix=initialize_triple_pointer_double(cropped->depth);

    uint32_t ind_d, ind_h, ind_w;

    for(ind_d=0;ind_d<cropped->depth;ind_d++){

        *(cropped->matrix+ind_d)=initialize_double_pointer_double(cropped->height);

        for(ind_h=0;ind_h<cropped->height;ind_h++){

                *(*(cropped->matrix+ind_d)+ind_h)=initialize_pointer_double(cropped->width);

                for(ind_w=0;ind_w<cropped->width;ind_w++){

                    *(*(*(cropped->matrix+ind_d)+ind_h)+ind_w)=to_be_cropped->matrix[ind_d][ind_h][ind_w];


            }
        }
    }

    return cropped;

}


void calculate_deltas__conv(Model** model, LAYER** layer){


    Blocks* weights=extract_from_Blocks((*layer)->kernels->blocks,0);

    //Block*

    Block* next_deltas=(*layer)->next_layer->deltas->block;
    //Block* next_cash=(*layer)->next_layer->cash->block;

    Block* cash=(*layer)->cash->block;

    uint32_t height_to_be_padded=cash->height-next_deltas->height+1;


    AddPadding_Block(&next_deltas,height_to_be_padded);


    Block* block_next_deltas=get_one_dimension_from_block(next_deltas,0);
    Block* block_first_kernel=Flip_Block(get_one_dimension_from_block(weights,0));
    Block* Z_i=get_one_dimension_from_block(cash,0);
    Block* next_cash=(*layer)->next_layer->cash->block;;

    //shape_block(block_first_kernel);
    //shape_block(block_next_deltas);

    write("next deltas");
    shape_block(block_next_deltas);

    write("first kernel");
    shape_block(block_first_kernel);
    //shape_block((*layer)->output_data->block);


    write("ok");
    Grid* output=convolve(block_next_deltas,block_first_kernel,1,0);

    Block* output_block;

    write("ok");
    get_Block_from_Grid(&output,&output_block);

    write("ok");
    Block* cropped_output_block=crop_Block(output_block,Z_i);

    write("ok");
    write("output conv :");
    shape_block(cropped_output_block);

    write("Z_i :");
    shape_block(Z_i);

    write("next cash :");
    shape_block(next_cash);



    /* the challenge now is to calculate the actual delta

    based on the weights, the next deltas and the output ;

    */

}

Block* mean_block(Block* block){
    /* Compress all the channels of a block into an average one channel block */


    Block* mean_block=initialize_Block(1);

    mean_block->depth=1;
    mean_block->height=block->height;
    mean_block->width=block->width;

    mean_block->matrix=initialize_triple_pointer_double(1);
    *(mean_block->matrix)=initialize_double_pointer_double(block->height);

    uint32_t ind_d,ind_h,ind_w;

    for(ind_h=0;ind_h<block->height;ind_h++){

        *(*(mean_block->matrix)+ind_h)=initialize_pointer_double(block->width);

        for(ind_w=0;ind_w<block->width;ind_w++){

            double sum=0.;

            for(ind_d=0;ind_d<block->depth;ind_d++){

                    sum+=block->matrix[ind_d][ind_h][ind_w];

                }
                *(*(*(mean_block->matrix)+ind_h)+ind_w)=sum/(block->depth);

        }
    }

    return mean_block;

}


void append_Block(Block** main_block, Block** block){


    if(!test_block_null_dimension(*main_block)) {

            *main_block=*block;

    }

    else {

    Block* new_main_block=initialize_Block(1);

    if((*main_block)->height!=(*block)->height || (*main_block)->width!=(*block)->width){

        ERROR_DIM_BLOCK_DIMENSIONS;
        exit(0);

    }

    new_main_block->depth=(*main_block)->depth+(*block)->depth;
    new_main_block->height=(*main_block)->height;
    new_main_block->width=(*main_block)->width;

    new_main_block->matrix=initialize_triple_pointer_double(new_main_block->depth);

    uint32_t ind_d,ind_h,ind_w;

    for(ind_d=0;ind_d<(*main_block)->depth;ind_d++){
        *(new_main_block->matrix+ind_d)=initialize_double_pointer_double((*main_block)->height);
        for(ind_h=0;ind_h<new_main_block->height;ind_h++){

            *(*(new_main_block->matrix+ind_d)+ind_h)=initialize_pointer_double((*main_block)->width);
            for(ind_w=0;ind_w<new_main_block->width;ind_w++){

                *(*(*(new_main_block->matrix+ind_d)+ind_h)+ind_w)=(*main_block)->matrix[ind_d][ind_h][ind_w];
            }
        }
    }


    for(ind_d=(*main_block)->depth;ind_d<new_main_block->depth;ind_d++){
        *(new_main_block->matrix+ind_d)=initialize_double_pointer_double((*block)->height);

        for(ind_h=0;ind_h<new_main_block->height;ind_h++){
            *(*(new_main_block->matrix+ind_d)+ind_h)=initialize_pointer_double((*block)->width);
            for(ind_w=0;ind_w<new_main_block->width;ind_w++){

                *(*(*(new_main_block->matrix+ind_d)+ind_h)+ind_w)=(*block)->matrix[ind_d-(*main_block)->depth][ind_h][ind_w];
            }
        }
    }

    *main_block=new_main_block;

    free(new_main_block);

    }

}


void summary_layers(Model** model,char* choice){


    if(choice=="forward"){

        LAYER* current=(*model)->first_layer;

        write("\n");
        while(current){
            write(getType(current->name));
            current=current->next_layer;
        }
    }
    else
    if(choice=="backward"){

        LAYER* current=(*model)->final_layer;

        write("\n");
        while(current){
            write(getType(current->name));
            current=current->previous_layer;
        }
    }
    else{
        write("Uknown choice : either 'forward' or 'backward'..");
    }

    printf("\n\nIn total we have %d layers \n",(*model)->nbr_levels);

}

void model_code(){


    Model* model;

    //declaring the input | output
    Block* X;

    create_Block(&X,4,179,179,"random","float");

    Grid* Y=fill_index(12,2);
    //create_Grid(&Y,5,5,"random","int");

    create_Model(&model,X,Y);


    add_CONV(&model,4,1,2,3,&relu);
    add_POOL(&model,2,2,3,"max");
    add_CONV(&model,5,1,1,7,&relu);
    add_POOL(&model,1,1,5,"max");
    add_CONV(&model,10,2,2,7,&relu);
    add_POOL(&model,2,2,3,"avg");
    add_CONV(&model,5,2,2,5,&relu);


    add_POOL(&model,1,1,5,"max");
    add_FLAT(&model);
    add_FCAF(&model,&tanh,100);
    add_FC(&model,&sigmoid,80);
    add_FC(&model,&sigmoid,60);
    add_FC(&model,&tanh,50);
    add_FC(&model,&tanh,20);
    add_FC(&model,&sigmoid,12);
    DENSE(&model);

    summary_layers(&model,"forward");


    LAYER* l_1=model->final_layer->previous_layer;
    LAYER* l=model->final_layer;
    LAYER* l_p=l_1->previous_layer->previous_layer->previous_layer->previous_layer->previous_layer;

    calculate_deltas_fc(&model,&l);
    calculate_deltas_fc(&model,&l_1);

    calculate_deltas_fc(&model,&(l_1->previous_layer));
    calculate_deltas_fc(&model,&(l_1->previous_layer->previous_layer));

    calculate_deltas_fc(&model,&(l_1->previous_layer->previous_layer->previous_layer));
    calculate_deltas_fc(&model,&(l_1->previous_layer->previous_layer->previous_layer->previous_layer));

    calculate_deltas_fc(&model,&(l_1->previous_layer->previous_layer->previous_layer->previous_layer->previous_layer));

    calculate_deltas_fc(&model,&(l_p->previous_layer));

    calculate_deltas__pool(&model,&(l_p->previous_layer->previous_layer));
    //transform_deltas_flatten(&model,&(l_1->previous_layer->previous_layer->previous_layer->previous_layer->previous_layer->previous_layer));

    calculate_deltas__conv(&model,&(l_p->previous_layer->previous_layer->previous_layer));

}


int main()
{

    //Debugging the code
    //model_code();

    Block* b0;
    Block* b1;
    Block* b=initialize_Block(1);

    create_Block(&b0,5,2,2,"random","float");
    create_Block(&b1,5,2,2,"random","float");

    b=element_wise_multiplication(b0,b1);

    //
    display_Block(b);
    //display_Block(b1);
    //display_Block(b);

    printf("\nDONE :))) ! \n\n");

    return 0;
}

