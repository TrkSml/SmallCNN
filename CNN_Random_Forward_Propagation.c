// __author__ = Tarek Samaali

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>

#define UPPER_BOUND .5
#define GEN_RANDOM_SEED srand(time(NULL))
#define DEBUG printf("debug !")
#define ERROR_DIMENSION_CONV printf("Dimension conflict: Please review the dimensions of the convolved arrays! \n")
#define ERROR_DEPTH printf("Cannot perform convolution: Please make sure the kernel and the block have the same depth. \n")
#define ERROR_NULL printf("Cannot perform convolution: Please make sure both the kernel and the block are not null. \n")
#define ERROR_CREATING_BLOCKS printf("Cannot create blocks : please make sure the length is > 1.\n")
#define ERROR_EVEN_DIMENSIONS printf(" How about using a kernel with odd sizes :))) ! .. ")
#define ERROR_DIMENSION_GRID_MULT printf("Please review the dimensions of the grid you want to perform multiplication on.\n")
#define ERROR_DIM_FLATTEN printf("Input must be flattened.\n")

#define current_Layer(x) printf("\nCurrent Layer: %s\n",x)
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))
#define min(X, Y)  ((X) < (Y) ? (X) : (Y))

//2D output
//After single convolution
typedef struct {
   unsigned int width;
   unsigned int height;
    float** grid;

} Grid;

//3D Block for a single
typedef struct {
   unsigned int depth;
   unsigned int width;
   unsigned int height;
    float*** matrix;
} Block;

//4D output
//After convolution with N filters
typedef struct{
   unsigned int length;
    Block** blocks;
}Blocks;

typedef struct{

    unsigned int previous_size;
    unsigned int current_size;
    float* bias;
    Grid* weights;
    Grid* Before_Activation;
    Grid* After_Activation;
    float (*activation)(float);

}FullyConnected;

float generate_random(){
    return ((float)rand())/((float)RAND_MAX) * UPPER_BOUND;

}

int control_parity_kernel_size(int size_kernel){
    return size_kernel%2==1;
}


int determine_size_output(int input_height,unsigned int kernel_height,unsigned int padding,unsigned int stride){

     return (int)(((input_height-kernel_height+2*padding)/stride))+1;
}

float relu(float x){
    return max(0,x);
}

float sigmoid(float x){
   return 1./(1.+exp(-x));
}


void shape_block(Block* block){

    printf("depth : %d \n",block->depth);
    printf("height : %d \n",block->height);
    printf("width : %d \n",block->width);

}

void shape_grid(Grid* grid){

    printf("\nheight : %d \n",grid->height);
    printf("width : %d \n",grid->width);

}


int test_block_null_dimension(Block* block){

    return block->height && block->width && block->depth;
}

int test_grid_null_dimension(Grid* grid){

    return grid->height && grid->width ;
}


int test_block_for_fully_connected(Block* block){

    return block->width==1 && block->height==1 ;
}

int test_for_grid_elementwise_multiplication(Grid* grid1, Grid* grid2){

    return grid1->height==grid2->height && grid1->width==grid2->width ;

}

int test_equal_grids_dimensions(Block* block1, Block* block2){

    return block1->depth==block2->depth || block1->width==block2->width ||\
                            block1->height==block2->height ;
}


void create_Grid(Grid** grid,unsigned int input_height,unsigned int input_width,char* choice){

    *grid=(Grid*)malloc(sizeof(Grid));
    (*grid)->height=input_height;
    (*grid)->width=input_width;


    if(choice=="random"){

        (*grid)->grid=(float**)malloc(input_height*sizeof(float*));
       unsigned int counter_height;

        for(counter_height=0;counter_height<input_height;counter_height++){
           unsigned int counter_width;
            float* row=(float*)malloc(input_width*(sizeof(float)));

            for(counter_width=0;counter_width<input_width;counter_width++){
                        *(row+counter_width)=(float)generate_random();
                }
            *((*grid)->grid+counter_height)=row;
        }


    }
    else if(choice=="zeros"){

       (*grid)->grid=(float**)malloc(input_height*sizeof(float*));
       unsigned int counter_height;

        for(counter_height=0;counter_height<input_height;counter_height++){
           unsigned int counter_width;
            float* row=(float*)malloc(input_width*(sizeof(float)));

            for(counter_width=0;counter_width<input_width;counter_width++){
                        *(row+counter_width)=0.0;
                }
            *((*grid)->grid+counter_height)=row;
        }

    }
}

Grid* transpose(Grid* grid_to_transpose){

    Grid* transposed_grid=(Grid*)malloc(sizeof(Grid));
    transposed_grid->height=grid_to_transpose->width;
    transposed_grid->width=grid_to_transpose->height;

    unsigned int index_heigth, index_width;
    transposed_grid->grid=(float**)malloc(transposed_grid->height*sizeof(float*));

    for(index_heigth=0;index_heigth<transposed_grid->height;index_heigth++){

        float* row=(float*)malloc(transposed_grid->width*sizeof(float));
        for(index_width=0;index_width<transposed_grid->width;index_width++){

            *(row+index_width)=grid_to_transpose->grid[index_width][index_heigth];
        }
        *(transposed_grid->grid+index_heigth)=row;
    }

    return transposed_grid;
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



void create_Block(Block** block,unsigned int input_depth,unsigned int input_height,unsigned int input_width,char* choice){

    *block=(Block*)malloc(sizeof(Block));
    (*block)->height=input_height;
    (*block)->width=input_width;
    (*block)->depth=input_depth;

    (*block)->matrix=(float***)malloc(input_depth*sizeof(float**));

   unsigned int index_depth;
    for(index_depth=0;index_depth<input_depth;index_depth++){

        Grid* grid;
        create_Grid(&grid,input_height,input_width,choice);
        *((*block)->matrix+index_depth)=grid->grid;

    }

}


void create_Blocks(Blocks **blocks,unsigned int length,unsigned int depth,unsigned int height,unsigned int width, char* choice){

    if(length<1){

        ERROR_CREATING_BLOCKS;
        exit(0);

    }else{

       *blocks=(Blocks *)malloc(sizeof(Blocks));
       Blocks* blocks_tmp=*blocks;
       blocks_tmp->blocks=malloc(length*sizeof(Block));
      unsigned int index_length;

       for(index_length=0;index_length<length;index_length++){
            Block *new_block=(Block*)malloc(sizeof(Block));
            create_Block(&new_block,depth,height,width,choice);
            //display_Block(new_block);
            *(blocks_tmp->blocks+index_length)=new_block;
       }

       blocks_tmp->length=length;

    }
}

Grid* extract_grid_from_given_depth(Block** block, unsigned int index_depth){

    Grid* current_depth=(Grid*)malloc(sizeof(Grid));

    (current_depth)->height=(*block)->height;
    (current_depth)->width=(*block)->width;
    (current_depth)->grid=(*block)->matrix[index_depth];

    return current_depth;


}


void apply_function_to_Grid(Grid** grid, float (*pointer_to_function)(float)){

    int index_width,index_height;

    for(index_height=0;index_height<(*grid)->height;index_height++){
        for(index_width=0;index_width<(*grid)->width;index_width++){
                float *current_element=&((*grid)->grid[index_height][index_width]);
                ((*grid)->grid[index_height][index_width])=(*pointer_to_function)(*current_element);


        }
    }

}



void apply_function_to_Block(Block** block, float (*pointer_to_function)(float)){

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
                         unsigned int begin_input_height,unsigned int end_input_height,\
                         unsigned int begin_input_width,unsigned int end_input_width){

    Grid* output_grid=(Grid*)malloc(sizeof(Grid));

   unsigned int size_height=end_input_height-begin_input_height;
   unsigned int size_width=end_input_width-begin_input_width;

    output_grid->width=size_width;
    output_grid->height=size_height;

    output_grid->grid=(float**)malloc(output_grid->height*sizeof(float*));

   unsigned int counter_height;
   unsigned int new_counter_height=0;

    for(counter_height=begin_input_height;counter_height<end_input_height;counter_height++){

       unsigned int counter_width;
       unsigned int new_counter_width=0;
        float* row=(float*)malloc(output_grid->width*(sizeof(float)));

        for(counter_width=begin_input_width;counter_width<end_input_width;counter_width++){
                    *(row+new_counter_width)=grid->grid[counter_height][counter_width];
                    new_counter_width++;

            }

        output_grid->grid[new_counter_height]=row;
        new_counter_height++;
    }

    return output_grid;

}


Block* Extract_From_Block(Block* grid,\
                         unsigned int begin_input_depth,unsigned int end_input_depth,\
                         unsigned int begin_input_height,unsigned int end_input_height,\
                         unsigned int begin_input_width,unsigned int end_input_width){


    Block* output_grid=(Block*)malloc(sizeof(Block));

   unsigned int size_depth=end_input_depth-begin_input_depth;
   unsigned int size_height=end_input_height-begin_input_height;
   unsigned int size_width=end_input_width-begin_input_width;

    output_grid->depth=size_depth;
    output_grid->width=size_width;
    output_grid->height=size_height;

    output_grid->matrix=(float***)malloc(size_depth*sizeof(float**));

   unsigned int counter_depth;
   unsigned int new_counter_depth=0;

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


Grid* AddPadding_Grid(Grid** block,int padding){

    Grid* output_grid;

   unsigned int height=(*block)->height;
   unsigned int width=(*block)->width;
    float **block_matrix=(*block)->grid;

    create_Grid(&output_grid,height+2*padding,width+2*padding,"zeros");

    //float** output_grid_matrix=output_grid->grid;

   unsigned int counter_height;
   unsigned int counter_width;


        for(counter_height=padding;counter_height<padding+height;counter_height++){

            for(counter_width=padding;counter_width<padding+width;counter_width++){

                output_grid->grid[counter_height][counter_width]=\
                    block_matrix[counter_height-padding][counter_width-padding];

            }
    }

    return output_grid;

}


void AddPadding_Block(Block** block,int padding){

    Block *output_Block=(Block*)malloc(sizeof(Block));

    output_Block->depth=(*block)->depth;

    output_Block->height=(*block)->height+2*padding;
    output_Block->width=(*block)->width+2*padding;
    output_Block->matrix=(float***)malloc(output_Block->depth*sizeof(float**));


   unsigned int counter_depth;

    for(counter_depth=0;counter_depth<output_Block->depth;counter_depth++){

            Grid* padded_grid=extract_grid_from_given_depth(block,counter_depth);

            *(output_Block->matrix+counter_depth)=AddPadding_Grid(&padded_grid,padding)->grid;

            free(padded_grid);


    }

    *block=output_Block;

}

float convolve_multiplication_sum(Block* block1, Block* block2){
   unsigned int depth,width,height;

    if(!test_equal_grids_dimensions(block1,block2)){

                ERROR_DIMENSION_CONV;
                exit(0);
                            }
    else{
        float output=0;

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

       unsigned int height=block->height;

        AddPadding_Block(&block,padding);

       unsigned int size_half_kernel=((kernel->height-1)/2);
       unsigned int begin_point_height=size_half_kernel;
       unsigned int end_point_height=block->height-begin_point_height;

       unsigned int begin_point_width=size_half_kernel;
       unsigned int end_point_width=block->width-begin_point_width;

       unsigned int index_height_output;
       unsigned int index_width_output;

        Grid* output_convolution_grid=(Grid*)malloc(sizeof(Grid));
        output_convolution_grid->height=(int)(end_point_height-begin_point_height)/stride;
        output_convolution_grid->width=(int)(end_point_width-begin_point_width)/stride;

        float** grid=(float**)malloc(output_convolution_grid->height*sizeof(float*));


        for(index_height_output=begin_point_height;index_height_output<end_point_height;index_height_output+=stride){

            float *row=(float*)malloc(output_convolution_grid->width*sizeof(float));

            for(index_width_output=begin_point_width;index_width_output<end_point_width;index_width_output+=stride){


                Block* extracted_block=Extract_From_Block(block,0,kernel->depth,index_height_output-size_half_kernel,\
                                                          index_height_output+size_half_kernel+1,index_width_output-size_half_kernel,\
                                                          index_width_output+size_half_kernel+1);


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

void Convolution(Block **input, Blocks * kernels,unsigned int stride,unsigned int padding){


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
        output->matrix=(float***)malloc(output->depth*sizeof(float**));


        // We have now to fill the output_matrix;

       unsigned int index_output_depth;
        for(index_output_depth=0;index_output_depth<output->depth;index_output_depth++){


            Grid* grid=convolve(*input, kernels->blocks[index_output_depth],stride,padding);
            *(output->matrix+index_output_depth)=grid->grid;
            free(grid);

        }

        *input=output;
    }

}

float Pooling_On_Extracted_Grid(Grid* block, char* choice){
   unsigned int width,height;

    if(choice=="max"){
        float output=0;

        for(height=0;height<block->height;height++){
            for(width=0;width<block->width;width++){

                output=max(output,block->grid[height][width]);

            }
        }

        return output;

    }else
    if(choice=="average"){

        //// Let us see
        float output=0;
        for(height=0;height<block->height;height++){
            for(width=0;width<block->width;width++){

                    output+=block->grid[height][width];

            }
        }

        return output/((block->height)*(block->width));

    }
    }


Grid* Pooling_On_Grid(Grid* grid,unsigned int size_kernel,unsigned int stride,unsigned int padding,char* choice){

    if(!test_grid_null_dimension(grid)){
        ERROR_NULL;
        exit(0);

    }else

    if(!control_parity_kernel_size(size_kernel)){
        ERROR_EVEN_DIMENSIONS;
        exit(0);

    }
    else {

   unsigned int height=grid->height;

    grid=AddPadding_Grid(&grid,padding);

    // We might as well add a size_output for the width

   unsigned int size_half_kernel=((size_kernel-1)/2);
   unsigned int begin_point_height=size_half_kernel;
   unsigned int end_point_height=grid->height-begin_point_height;

   unsigned int begin_point_width=size_half_kernel;
   unsigned int end_point_width=grid->width-begin_point_width;

   unsigned int index_height_output;
   unsigned int index_width_output;

    Grid* output_convolution_grid=(Grid*)malloc(sizeof(Grid));
    output_convolution_grid->height=(end_point_height-begin_point_height)/stride;
    output_convolution_grid->width=(end_point_width-begin_point_width)/stride;

    output_convolution_grid->grid=(float**)malloc(output_convolution_grid->height*sizeof(float*));


    for(index_height_output=begin_point_height;index_height_output<end_point_height;index_height_output++){
        float *row=(float*)malloc(output_convolution_grid->width*sizeof(float));

        for(index_width_output=begin_point_width;index_width_output<end_point_width;index_width_output++){

            Grid* extracted_grid=Extract_From_Grid(grid,index_height_output-size_half_kernel,\
                                                      index_height_output+size_half_kernel+1,index_width_output-size_half_kernel,\
                                                      index_width_output+size_half_kernel+1);


            *(row+(index_width_output-begin_point_width)/stride)=Pooling_On_Extracted_Grid(extracted_grid,choice);

        }
        *(output_convolution_grid->grid+(index_height_output-begin_point_height)/stride)=row;
    }



    return output_convolution_grid;

    }
}

// We will continue at this level
void Pooling(Block **input,unsigned int size_kernel,unsigned int stride,unsigned int padding, char* choice){

    current_Layer("Pooling");

    if(!test_block_null_dimension(*input)){
        ERROR_NULL;
        exit(0);
    }
    else
    {

    Block* output=(Block*)malloc(sizeof(Block));
    output->height=determine_size_output((*input)->height, size_kernel, padding, stride);
    output->width=determine_size_output((*input)->width, size_kernel, padding, stride);

    output->depth=(*input)->depth;
    output->matrix=(float***)malloc(output->depth*sizeof(float**));

    // We have now to fill the output_matrix;


   unsigned int index_output_depth;
    for(index_output_depth=0;index_output_depth<output->depth;index_output_depth++){

        Grid* grid_from_current_block=extract_grid_from_given_depth(input,index_output_depth);
        Grid* pooled_grid=Pooling_On_Grid(grid_from_current_block,size_kernel,stride,padding,choice);

        *(output->matrix+index_output_depth)=pooled_grid->grid;

        free(grid_from_current_block);
        free(pooled_grid);

    }

    *input=output;

    }
}


// Take a block and decrease both its width and height to 1
void extract_Grid_From_Flatten_Block(Block** block, Grid** grid){

    *grid=(Grid*)malloc(sizeof(Grid));
    (*grid)->height=(*block)->height;
    (*grid)->width=(*block)->depth;

    (*grid)->grid=(float**)malloc(sizeof(float*));
    float* row=(float*)malloc((*grid)->width*sizeof(float));

    unsigned int index_depth;
    for(index_depth=0;index_depth<(*grid)->width;index_depth++){

        *(row+index_depth)=(*block)->matrix[index_depth][0][0];
    }

    *((*grid)->grid)=row;

}


void Flatten(Block **input){

    //Block* block=*(input);
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

    float*** Flattened=(float***)malloc(block->depth*sizeof(float**));

   unsigned int index_depth;
   unsigned int index_height;
   unsigned int index_width;

   unsigned int counter_array_flattened=0;

    for(index_depth=0;index_depth<(*input)->depth;index_depth++){
            for(index_height=0;index_height<(*input)->height;index_height++){
                for(index_width=0;index_width<(*input)->width;index_width++){

                    float** decrease_height=(float**)malloc(sizeof(float*));
                    float* decrease_width=(float*)malloc(sizeof(float));

                    *decrease_width=(*input)->matrix[index_depth][index_height][index_width];
                    *decrease_height=decrease_width;


                    Flattened[counter_array_flattened]=decrease_height;

                    counter_array_flattened++;

            }

        }

    }

    block->matrix=Flattened;
    *input=block;

    }

}

void Stack_Blocks(int nbr_arguments,... ){

   unsigned int i;
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

        (*output_grid)->grid=(float**)malloc((*output_grid)->height*sizeof(float*));


        int index_height, index_width;

        for(index_height=0;index_height<(*output_grid)->height;index_height++){


            float* row=(float*)malloc((*output_grid)->height*sizeof(float));

            for(index_width=0;index_width<(*output_grid)->width;index_width++){


                int index_forgotten;
                float sum_forgotten=0.0;

                for(index_forgotten=0;index_forgotten<(*grid1)->width;index_forgotten++){
                    sum_forgotten+=(*grid1)->grid[index_height][index_forgotten]*((*grid2)->grid[index_forgotten][index_width]);

                }
                *(row+index_width)=sum_forgotten;

            }
            *((*output_grid)->grid+index_height)=row;

            }
        }

}


void grid_element_wise_mutiplication(Grid** output_grid, Grid** grid1, Grid** grid2){

    if(!test_for_grid_elementwise_multiplication(*grid1,*grid2)){

        ERROR_DIMENSION_GRID_MULT;
        exit(0);

    }
    else{

        *output_grid=(Grid*)malloc(sizeof(Grid));
        (*output_grid)->height=(*grid1)->height;
        (*output_grid)->width=(*grid1)->width;
        (*output_grid)->grid=(float**)malloc((*output_grid)->height*sizeof(float*));

        int index_height, index_width;

        for(index_height=0;index_height<(*output_grid)->height;index_height++){

            float* row=(float*)malloc((*output_grid)->height*sizeof(float));

            for(index_width=0;index_width<(*output_grid)->width;index_width++){

                *(row+index_width)=(*grid1)->grid[index_height][index_width]*((*grid2)->grid[index_height][index_width]);

            }
            *((*output_grid)->grid+index_height)=row;

        }

    }

}

// Creating the Fully connected layers

Grid copy_grid(Grid to_fill, Grid to_copy){

    //*to_fill=(Grid*)malloc(sizeof(Grid));
    printf("%x\n\n",&to_fill);
    printf("%x\n\n",&to_copy);
    to_fill.width=(to_copy).width;
    to_fill.height=to_copy.height;
    to_fill.grid=(to_copy).grid;


    return to_fill;

}


void Fully_Connected_After_Flatten(FullyConnected** fc, Block** input, float (*activation)(float), int output_layer_size){

    if(!*fc){

        if(!test_block_for_fully_connected(*input)){

            ERROR_DIM_FLATTEN ;
            exit(0);

        }else

        {

            *fc=(FullyConnected*)malloc(sizeof(FullyConnected));;
            FullyConnected* local_fc=*fc;

            unsigned int input_layer_size=(*input)->depth;

            local_fc->bias=calloc(output_layer_size,sizeof(float));

            Grid* weights_tmp;
            create_Grid(&weights_tmp,output_layer_size,input_layer_size,"random");

            local_fc->weights=weights_tmp;
            local_fc->activation=*activation;

            Grid* Z_i;
            Grid* A_i;

            Grid* input_grid;

            extract_Grid_From_Flatten_Block(input,&input_grid);
            Grid* transposed_input_grid=transpose(input_grid);

            grid_dot_mutiplication(&Z_i,&local_fc->weights,&transposed_input_grid);
            grid_dot_mutiplication(&A_i,&local_fc->weights,&transposed_input_grid);

            local_fc->Before_Activation=Z_i;

            apply_function_to_Grid(&A_i,local_fc->activation);

            local_fc->After_Activation=A_i;
            local_fc->previous_size=input_layer_size;
            local_fc->current_size=output_layer_size;

            //Add bias


            }
        }

    else{


    // Treat the case where fc is not null: meaning that it is not the first time


    }
}

void display_Block(Block* grid){

   unsigned int dpth,row,col;

    for(dpth=0;dpth<grid->depth;dpth++){
        printf("Level : %d\n",dpth+1);
        for(row=0;row<grid->height;row++){
            for(col=0;col<grid->width;col++){
                printf("%.2f |", grid->matrix[dpth][row][col]);
            }
            printf("\n");
        }
        printf(" \n \n");
    }
    printf("\n");
}


void display_Grid(Grid *table){
   unsigned int row,col;

    for(row=0;row<table->height;row++){
        for(col=0;col<table->width;col++){
            printf("%.5f |", table->grid[row][col]);
            }
        printf("\n");
    }

    printf("\n");

}

void debug_code(){

    //Creating random input

    Block* input;
    create_Block(&input,5,10,10,"random");

    //Creating random kernels
    Blocks* kernels;
    create_Blocks(&kernels,5,5,3,3,"random");

    //Covolution Layer
    Convolution(&input,kernels,1,1);
    shape_block(input);

    //Pooling Layer
    Pooling(&input,3,2,1,"max");
    shape_block(input);

    Blocks* kernels_bis;
    create_Blocks(&kernels_bis,15,5,5,5,"random");

    //Covolution Layer
    Convolution(&input,kernels_bis,1,2);
    shape_block(input);

    //Pooling Layer
    Pooling(&input,5,1,0,"max");
    shape_block(input);


    Flatten(&input);
    shape_block(input);

    FullyConnected* fc=NULL;
    Fully_Connected_After_Flatten(&fc,&input,&sigmoid,20);

    //Display Input

}

void print(int a){
    printf("\n%d\n",a);
}
int main()
{

    //Debugging the code
    debug_code();


    printf("\nDONE :))) ! \n\n");

    return 0;
}

